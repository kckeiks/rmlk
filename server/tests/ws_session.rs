//! WebSocket session integration tests (shared client helper).

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rmlk_server::http::{new_app_state, new_app_state_with_drop_counter};
use rmlk_server::protocol::{error_code, error_message, ClientFrame, ServerFrame};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use common::TestServer;

#[tokio::test]
async fn rejects_garbage_first_frame() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut client = server.connect().await;

    client.send_binary(vec![0xff, 0, 0, 0, 0]).await;
    assert_eq!(
        client.recv_frame().await,
        ServerFrame::Error {
            code: error_code::MALFORMED_FRAME,
            message: error_message::MALFORMED_FRAME.into(),
        }
    );
    assert!(matches!(client.recv_raw().await, WsMessage::Close(_)));

    server.shutdown().await;
}

#[tokio::test]
async fn rejects_non_open_first_frame() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut client = server.connect().await;

    client
        .send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    assert_eq!(
        client.recv_frame().await,
        ServerFrame::Error {
            code: error_code::UNEXPECTED_FRAME,
            message: error_message::EXPECTED_OPEN.into(),
        }
    );
    assert!(matches!(client.recv_raw().await, WsMessage::Close(_)));

    server.shutdown().await;
}

#[tokio::test]
async fn open_returns_open_ack() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut client = server.connect().await;

    let _id = client.open_session().await;
    client.close().await;
    server.shutdown().await;
}

#[tokio::test]
async fn audio_returns_partial() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut client = server.connect().await;

    client.open_session().await;
    client
        .send_frame(&ClientFrame::Audio {
            pcm16: vec![0, 1, 2],
        })
        .await;
    assert_eq!(
        client.recv_frame().await,
        ServerFrame::Partial {
            text: "partial-1".into()
        }
    );

    client.close().await;
    server.shutdown().await;
}

#[tokio::test]
async fn finalize_returns_final_and_closes() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut client = server.connect().await;

    client.open_session().await;
    client
        .send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    assert!(matches!(
        client.recv_frame().await,
        ServerFrame::Partial { .. }
    ));

    client.send_frame(&ClientFrame::Finalize).await;
    assert_eq!(
        client.recv_frame().await,
        ServerFrame::Final {
            text: "final-1".into()
        }
    );
    assert!(matches!(client.recv_raw().await, WsMessage::Close(_)));

    server.shutdown().await;
}

#[tokio::test]
async fn disconnect_frees_session() {
    let drops = Arc::new(AtomicUsize::new(0));
    let state = new_app_state_with_drop_counter(Arc::clone(&drops));
    let server = TestServer::spawn(state.clone()).await;
    let mut client = server.connect().await;

    client.open_session().await;
    assert_eq!(server.state.live_session_count().await, 1);

    client
        .send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    let _ = client.recv_frame().await;

    client.close().await;

    tokio::time::timeout(Duration::from_secs(2), async {
        while server.state.live_session_count().await != 0 || drops.load(Ordering::SeqCst) != 1
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("session should be freed after client disconnect");

    assert_eq!(server.state.live_session_count().await, 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    server.shutdown().await;
}