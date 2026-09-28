//! HTTP routes served by the binary.

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::{routing::get, Router};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use crate::engine::{EngineEvent, MockEngine};
use crate::protocol::{error_code, error_message, ClientFrame, ServerFrame};
use crate::session::{SessionError, SessionRegistry, StreamState};

#[derive(Clone)]
pub struct AppState {
    registry: Arc<Mutex<SessionRegistry>>,
    engine: Arc<Mutex<MockEngine>>,
}

/// Application router with shared registry + engine.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
}

fn new_app_state() -> AppState {
    AppState {
        registry: Arc::new(Mutex::new(SessionRegistry::default())),
        engine: Arc::new(Mutex::new(MockEngine::new())),
    }
}

async fn health() -> &'static str {
    "ok"
}

async fn ws_upgrade(State(state): State<AppState>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let Some(first) = recv_binary_frame(&mut socket).await else {
        return;
    };

    match ClientFrame::decode(&first) {
        Ok(ClientFrame::Open) => {
            let (id, mut stream) = state.registry.lock().await.open();
            if send_frame(
                &mut socket,
                &ServerFrame::OpenAck {
                    session_id: id.as_u64(),
                },
            )
            .await
            .is_err()
            {
                let _ = state.registry.lock().await.unregister(id);
                return;
            }
            run_session(&mut socket, &mut stream, &state.engine).await;
            let _ = state.registry.lock().await.unregister(id);
        }
        Ok(_) => {
            send_error_and_close(
                &mut socket,
                error_code::UNEXPECTED_FRAME,
                error_message::EXPECTED_OPEN,
            )
            .await;
        }
        Err(_) => {
            send_error_and_close(
                &mut socket,
                error_code::MALFORMED_FRAME,
                error_message::MALFORMED_FRAME,
            )
            .await;
        }
    }
}

/// Connection-owned audio loop: engine lock only around push, never the registry.
async fn run_session(
    socket: &mut WebSocket,
    stream: &mut StreamState,
    engine: &Arc<Mutex<MockEngine>>,
) {
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Close(_) => break,
            Message::Binary(bytes) => match ClientFrame::decode(&bytes) {
                Ok(ClientFrame::Audio { pcm16 }) => {
                    let events = {
                        let mut engine = engine.lock().await;
                        match stream.push_audio(&mut *engine, &pcm16) {
                            Ok(events) => events,
                            Err(SessionError::Busy) => {
                                send_error_and_close(
                                    socket,
                                    error_code::BUSY,
                                    error_message::BUSY,
                                )
                                .await;
                                break;
                            }
                            Err(_) => {
                                send_error_and_close(
                                    socket,
                                    error_code::INTERNAL,
                                    error_message::INTERNAL,
                                )
                                .await;
                                break;
                            }
                        }
                    };
                    if send_engine_events(socket, events).await.is_err() {
                        break;
                    }
                }
                Ok(ClientFrame::Finalize) | Ok(ClientFrame::Cancel) => {
                    // Handled in later checklist items.
                    break;
                }
                Ok(ClientFrame::Open) => {
                    send_error_and_close(
                        socket,
                        error_code::UNEXPECTED_FRAME,
                        error_message::EXPECTED_OPEN,
                    )
                    .await;
                    break;
                }
                Err(_) => {
                    send_error_and_close(
                        socket,
                        error_code::MALFORMED_FRAME,
                        error_message::MALFORMED_FRAME,
                    )
                    .await;
                    break;
                }
            },
            _ => {}
        }
    }
}

async fn send_engine_events(
    socket: &mut WebSocket,
    events: Vec<EngineEvent>,
) -> Result<(), ()> {
    for event in events {
        let frame = match event {
            EngineEvent::Partial { text } => ServerFrame::Partial { text },
            EngineEvent::Final { text } => ServerFrame::Final { text },
        };
        send_frame(socket, &frame).await?;
    }
    Ok(())
}

async fn recv_binary_frame(socket: &mut WebSocket) -> Option<axum::body::Bytes> {
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Binary(bytes) => return Some(bytes),
            Message::Close(_) => return None,
            _ => {}
        }
    }
    None
}

async fn send_frame(socket: &mut WebSocket, frame: &ServerFrame) -> Result<(), ()> {
    let bytes = frame.encode().map_err(|_| ())?;
    socket
        .send(Message::Binary(bytes.into()))
        .await
        .map_err(|_| ())
}

async fn send_error_and_close(socket: &mut WebSocket, code: u16, message: &str) {
    let _ = send_frame(
        socket,
        &ServerFrame::Error {
            code,
            message: message.into(),
        },
    )
    .await;
    let _ = socket.send(Message::Close(None)).await;
}

/// Serve `router` until `shutdown` completes.
pub async fn serve(
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(listener, router(new_app_state()))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

/// Completes when the process receives Ctrl-C.
pub async fn shutdown_on_ctrl_c() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl-C handler");
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use futures_util::{SinkExt, StreamExt};
    use http_body_util::BodyExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_tungstenite::tungstenite::Message as WsMessage;
    use tower::ServiceExt;

    use super::{new_app_state, router, serve};
    use crate::protocol::{error_code, error_message, ClientFrame, ServerFrame};

    async fn spawn_server() -> (std::net::SocketAddr, oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let _ = serve(listener, async {
                let _ = shutdown_rx.await;
            })
            .await;
        });
        (addr, shutdown_tx, server)
    }

    async fn open_session(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) {
        let open = ClientFrame::Open.encode().unwrap();
        ws.send(WsMessage::Binary(open.into())).await.unwrap();
        let reply = ws.next().await.unwrap().unwrap();
        let WsMessage::Binary(bytes) = reply else {
            panic!("expected OpenAck, got {reply:?}");
        };
        assert!(matches!(
            ServerFrame::decode(&bytes).unwrap(),
            ServerFrame::OpenAck { .. }
        ));
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let response = router(new_app_state())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn graceful_shutdown_stops_serving() {
        let (addr, shutdown_tx, server) = spawn_server().await;

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut buf = vec![0u8; 1024];
        let n = stream.read(&mut buf).await.unwrap();
        let response = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(response.contains("200"), "{response}");
        assert!(response.contains("ok"), "{response}");

        shutdown_tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_rejects_garbage_first_frame() {
        let (addr, shutdown_tx, server) = spawn_server().await;
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .unwrap();

        ws.send(WsMessage::Binary(vec![0xff, 0, 0, 0, 0].into()))
            .await
            .unwrap();

        let reply = ws.next().await.unwrap().unwrap();
        let WsMessage::Binary(bytes) = reply else {
            panic!("expected binary Error frame, got {reply:?}");
        };
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap(),
            ServerFrame::Error {
                code: error_code::MALFORMED_FRAME,
                message: error_message::MALFORMED_FRAME.into(),
            }
        );

        let close = ws.next().await.unwrap().unwrap();
        assert!(matches!(close, WsMessage::Close(_)), "{close:?}");

        shutdown_tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_rejects_non_open_first_frame() {
        let (addr, shutdown_tx, server) = spawn_server().await;
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .unwrap();

        let audio = ClientFrame::Audio { pcm16: vec![0] }
            .encode()
            .unwrap();
        ws.send(WsMessage::Binary(audio.into())).await.unwrap();

        let reply = ws.next().await.unwrap().unwrap();
        let WsMessage::Binary(bytes) = reply else {
            panic!("expected binary Error frame, got {reply:?}");
        };
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap(),
            ServerFrame::Error {
                code: error_code::UNEXPECTED_FRAME,
                message: error_message::EXPECTED_OPEN.into(),
            }
        );

        let close = ws.next().await.unwrap().unwrap();
        assert!(matches!(close, WsMessage::Close(_)), "{close:?}");

        shutdown_tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_open_returns_open_ack() {
        let (addr, shutdown_tx, server) = spawn_server().await;
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .unwrap();

        open_session(&mut ws).await;

        ws.close(None).await.unwrap();
        shutdown_tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_audio_returns_partial() {
        let (addr, shutdown_tx, server) = spawn_server().await;
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .unwrap();

        open_session(&mut ws).await;

        let audio = ClientFrame::Audio {
            pcm16: vec![0, 1, 2],
        }
        .encode()
        .unwrap();
        ws.send(WsMessage::Binary(audio.into())).await.unwrap();

        let reply = ws.next().await.unwrap().unwrap();
        let WsMessage::Binary(bytes) = reply else {
            panic!("expected binary Partial, got {reply:?}");
        };
        assert_eq!(
            ServerFrame::decode(&bytes).unwrap(),
            ServerFrame::Partial {
                text: "partial-1".into()
            }
        );

        ws.close(None).await.unwrap();
        shutdown_tx.send(()).unwrap();
        server.await.unwrap();
    }
}
