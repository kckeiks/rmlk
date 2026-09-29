//! Shared helpers for `rmlk-server` integration tests.
//!
//! Uses the same `protocol` codec as the server — not a second wire format.

pub mod corpus;

use std::net::SocketAddr;

use futures_util::{SinkExt, StreamExt};
use rmlk_server::http::{serve_with_state, AppState};
use rmlk_server::protocol::{ClientFrame, ServerFrame};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::WebSocketStream;

type WsStream = WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Running test server plus a connected WebSocket client.
pub struct TestServer {
    pub addr: SocketAddr,
    pub state: AppState,
    shutdown_tx: oneshot::Sender<()>,
    server: tokio::task::JoinHandle<()>,
}

impl TestServer {
    /// Bind on an ephemeral port and serve `state`.
    pub async fn spawn(state: AppState) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let serve_state = state.clone();
        let server = tokio::spawn(async move {
            let _ = serve_with_state(
                listener,
                async {
                    let _ = shutdown_rx.await;
                },
                serve_state,
            )
            .await;
        });
        Self {
            addr,
            state,
            shutdown_tx,
            server,
        }
    }

    /// Connect a WebSocket client to `/ws`.
    pub async fn connect(&self) -> WsClient {
        let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{}/ws", self.addr))
            .await
            .unwrap();
        WsClient { ws }
    }

    /// Signal shutdown and join the server task.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        let _ = self.server.await;
    }
}

/// Thin WebSocket client that speaks the server binary protocol.
pub struct WsClient {
    ws: WsStream,
}

impl WsClient {
    /// Send a protocol client frame as one binary WebSocket message.
    pub async fn send_frame(&mut self, frame: &ClientFrame) {
        let bytes = frame.encode().expect("encode client frame");
        self.ws
            .send(WsMessage::Binary(bytes.into()))
            .await
            .expect("send ws binary");
    }

    /// Send raw binary bytes (for garbage / malformed tests).
    pub async fn send_binary(&mut self, bytes: Vec<u8>) {
        self.ws
            .send(WsMessage::Binary(bytes.into()))
            .await
            .expect("send ws binary");
    }

    /// Receive and decode the next binary server frame.
    pub async fn recv_frame(&mut self) -> ServerFrame {
        let msg = self.ws.next().await.expect("ws stream").expect("ws message");
        let WsMessage::Binary(bytes) = msg else {
            panic!("expected binary server frame, got {msg:?}");
        };
        ServerFrame::decode(&bytes).expect("decode server frame")
    }

    /// Receive the next WebSocket message (e.g. Close).
    pub async fn recv_raw(&mut self) -> WsMessage {
        self.ws.next().await.expect("ws stream").expect("ws message")
    }

    /// `Open` then expect `OpenAck`.
    pub async fn open_session(&mut self) -> u64 {
        self.send_frame(&ClientFrame::Open).await;
        match self.recv_frame().await {
            ServerFrame::OpenAck { session_id } => session_id,
            other => panic!("expected OpenAck, got {other:?}"),
        }
    }

    /// Send a WebSocket Close and drop the client.
    pub async fn close(mut self) {
        self.ws.close(None).await.expect("ws close");
    }
}
