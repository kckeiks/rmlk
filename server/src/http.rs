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

use crate::engine::MockEngine;
use crate::protocol::{error_code, error_message, ClientFrame, ServerFrame};
use crate::session::Sessions;

type SessionStore = Arc<Mutex<Sessions<MockEngine>>>;

/// Application router with shared session store.
pub fn router(sessions: SessionStore) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_upgrade))
        .with_state(sessions)
}

fn new_session_store() -> SessionStore {
    Arc::new(Mutex::new(Sessions::new(MockEngine::new())))
}

async fn health() -> &'static str {
    "ok"
}

async fn ws_upgrade(State(sessions): State<SessionStore>, ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, sessions))
}

async fn handle_socket(mut socket: WebSocket, sessions: SessionStore) {
    let Some(first) = recv_binary_frame(&mut socket).await else {
        return;
    };

    match ClientFrame::decode(&first) {
        Ok(ClientFrame::Open) => {
            let id = sessions.lock().await.open();
            if send_frame(
                &mut socket,
                &ServerFrame::OpenAck {
                    session_id: id.as_u64(),
                },
            )
            .await
            .is_err()
            {
                let _ = sessions.lock().await.close(id);
                return;
            }
            drain_until_close(&mut socket).await;
            let _ = sessions.lock().await.close(id);
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

async fn drain_until_close(socket: &mut WebSocket) {
    while let Some(Ok(msg)) = socket.recv().await {
        if matches!(msg, Message::Close(_)) {
            break;
        }
    }
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
    axum::serve(listener, router(new_session_store()))
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

    use super::{new_session_store, router, serve};
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

    #[tokio::test]
    async fn health_returns_ok() {
        let response = router(new_session_store())
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

        // Unknown type tag — not a valid client frame.
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

        // Server should close afterward.
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

        let open = ClientFrame::Open.encode().unwrap();
        ws.send(WsMessage::Binary(open.into())).await.unwrap();

        let reply = ws.next().await.unwrap().unwrap();
        let WsMessage::Binary(bytes) = reply else {
            panic!("expected binary OpenAck, got {reply:?}");
        };
        match ServerFrame::decode(&bytes).unwrap() {
            ServerFrame::OpenAck { session_id: _ } => {}
            other => panic!("expected OpenAck, got {other:?}"),
        }

        ws.close(None).await.unwrap();
        shutdown_tx.send(()).unwrap();
        server.await.unwrap();
    }
}
