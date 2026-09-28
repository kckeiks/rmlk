//! HTTP routes served by the binary.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

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
    /// When set, each opened stream increments this on drop (tests).
    drop_counter: Option<Arc<AtomicUsize>>,
}

impl AppState {
    /// Number of live sessions in the registry.
    pub async fn live_session_count(&self) -> usize {
        self.registry.lock().await.len()
    }
}

/// Application router with shared registry + engine.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
}

/// Default in-process app state (mock engine).
pub fn new_app_state() -> AppState {
    AppState {
        registry: Arc::new(Mutex::new(SessionRegistry::default())),
        engine: Arc::new(Mutex::new(MockEngine::new())),
        drop_counter: None,
    }
}

/// App state that counts `StreamState` drops (integration tests).
pub fn new_app_state_with_drop_counter(drop_counter: Arc<AtomicUsize>) -> AppState {
    AppState {
        registry: Arc::new(Mutex::new(SessionRegistry::default())),
        engine: Arc::new(Mutex::new(MockEngine::new())),
        drop_counter: Some(drop_counter),
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
            if let Some(counter) = &state.drop_counter {
                stream.track_drops(Arc::clone(counter));
            }
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
            run_session(&mut socket, stream, &state.engine).await;
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

/// Connection-owned session loop: engine lock only around push/finalize, never the registry.
async fn run_session(
    socket: &mut WebSocket,
    mut stream: StreamState,
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
                                return;
                            }
                            Err(_) => {
                                send_error_and_close(
                                    socket,
                                    error_code::INTERNAL,
                                    error_message::INTERNAL,
                                )
                                .await;
                                return;
                            }
                        }
                    };
                    if send_engine_events(socket, events).await.is_err() {
                        return;
                    }
                }
                Ok(ClientFrame::Finalize) => {
                    let events = {
                        let mut engine = engine.lock().await;
                        match stream.finalize(&mut *engine) {
                            Ok(events) => events,
                            Err(_) => {
                                send_error_and_close(
                                    socket,
                                    error_code::INTERNAL,
                                    error_message::INTERNAL,
                                )
                                .await;
                                return;
                            }
                        }
                    };
                    let _ = send_engine_events(socket, events).await;
                    let _ = socket.send(Message::Close(None)).await;
                    return;
                }
                Ok(ClientFrame::Cancel) => {
                    let _ = socket.send(Message::Close(None)).await;
                    return;
                }
                Ok(ClientFrame::Open) => {
                    send_error_and_close(
                        socket,
                        error_code::UNEXPECTED_FRAME,
                        error_message::EXPECTED_OPEN,
                    )
                    .await;
                    return;
                }
                Err(_) => {
                    send_error_and_close(
                        socket,
                        error_code::MALFORMED_FRAME,
                        error_message::MALFORMED_FRAME,
                    )
                    .await;
                    return;
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
    serve_with_state(listener, shutdown, new_app_state()).await
}

/// Serve with an explicit shared [`AppState`] (tests / custom wiring).
pub async fn serve_with_state(
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
    state: AppState,
) -> Result<()> {
    axum::serve(listener, router(state))
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
    use http_body_util::BodyExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tower::ServiceExt;

    use super::{new_app_state, router, serve_with_state};

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
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let _ = serve_with_state(
                listener,
                async {
                    let _ = shutdown_rx.await;
                },
                new_app_state(),
            )
            .await;
        });

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
}
