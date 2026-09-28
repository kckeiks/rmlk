//! HTTP routes served by the binary.

use std::future::Future;

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::{routing::get, Router};
use tokio::net::TcpListener;

use crate::protocol::{error_code, error_message, ClientFrame, ServerFrame};

/// Application router.
pub fn router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_upgrade))
}

async fn health() -> &'static str {
    "ok"
}

async fn ws_upgrade(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    let Some(first) = recv_binary_frame(&mut socket).await else {
        return;
    };

    match ClientFrame::decode(&first) {
        Ok(ClientFrame::Open) => {
            // Session open / OpenAck lands in the next checklist item.
            while let Some(Ok(msg)) = socket.recv().await {
                if matches!(msg, Message::Close(_)) {
                    break;
                }
            }
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

async fn send_error_and_close(socket: &mut WebSocket, code: u16, message: &str) {
    let frame = ServerFrame::Error {
        code,
        message: message.into(),
    };
    if let Ok(bytes) = frame.encode() {
        let _ = socket.send(Message::Binary(bytes.into())).await;
    }
    let _ = socket.send(Message::Close(None)).await;
}

/// Serve `router` until `shutdown` completes.
pub async fn serve(
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    axum::serve(listener, router())
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

    use super::{router, serve};
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
        let response = router()
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
}
