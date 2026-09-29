//! WebSocket session integration tests (mock engine).
//!
//! See `server/docs/tests.md`.

mod utils;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rmlk_server::http::{new_app_state, new_app_state_with_drop_counter};
use rmlk_server::protocol::{error_code, error_message, ClientFrame, ServerFrame};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use utils::TestServer;

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

#[tokio::test]
async fn two_clients_distinct_ids_and_finals() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut a = server.connect().await;
    let mut b = server.connect().await;

    let id_a = a.open_session().await;
    let id_b = b.open_session().await;
    assert_ne!(id_a, id_b);
    assert_eq!(server.state.live_session_count().await, 2);

    a.send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    assert_eq!(
        a.recv_frame().await,
        ServerFrame::Partial {
            text: "partial-1".into()
        }
    );

    b.send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    assert_eq!(
        b.recv_frame().await,
        ServerFrame::Partial {
            text: "partial-1".into()
        }
    );
    b.send_frame(&ClientFrame::Audio { pcm16: vec![1] })
        .await;
    assert_eq!(
        b.recv_frame().await,
        ServerFrame::Partial {
            text: "partial-2".into()
        }
    );

    a.send_frame(&ClientFrame::Finalize).await;
    assert_eq!(
        a.recv_frame().await,
        ServerFrame::Final {
            text: "final-1".into()
        }
    );
    assert!(matches!(a.recv_raw().await, WsMessage::Close(_)));

    b.send_frame(&ClientFrame::Finalize).await;
    assert_eq!(
        b.recv_frame().await,
        ServerFrame::Final {
            text: "final-2".into()
        }
    );
    assert!(matches!(b.recv_raw().await, WsMessage::Close(_)));

    tokio::time::timeout(Duration::from_secs(2), async {
        while server.state.live_session_count().await != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("both sessions should unregister");

    server.shutdown().await;
}

#[tokio::test]
async fn cancel_on_a_does_not_affect_b() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut a = server.connect().await;
    let mut b = server.connect().await;

    a.open_session().await;
    b.open_session().await;
    assert_eq!(server.state.live_session_count().await, 2);

    a.send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    let _ = a.recv_frame().await;
    a.send_frame(&ClientFrame::Cancel).await;
    assert!(matches!(a.recv_raw().await, WsMessage::Close(_)));

    tokio::time::timeout(Duration::from_secs(2), async {
        while server.state.live_session_count().await != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("A should unregister after cancel");

    // B's chunk count is independent (still at 0 pushes so far).
    b.send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    assert_eq!(
        b.recv_frame().await,
        ServerFrame::Partial {
            text: "partial-1".into()
        }
    );
    b.send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    assert_eq!(
        b.recv_frame().await,
        ServerFrame::Partial {
            text: "partial-2".into()
        }
    );
    b.send_frame(&ClientFrame::Finalize).await;
    assert_eq!(
        b.recv_frame().await,
        ServerFrame::Final {
            text: "final-2".into()
        }
    );
    assert!(matches!(b.recv_raw().await, WsMessage::Close(_)));

    tokio::time::timeout(Duration::from_secs(2), async {
        while server.state.live_session_count().await != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("B should unregister after finalize");

    server.shutdown().await;
}

#[tokio::test]
async fn disconnect_on_a_does_not_affect_b() {
    let server = TestServer::spawn(new_app_state()).await;
    let mut a = server.connect().await;
    let mut b = server.connect().await;

    a.open_session().await;
    b.open_session().await;

    a.send_frame(&ClientFrame::Audio { pcm16: vec![0] })
        .await;
    let _ = a.recv_frame().await;
    a.close().await;

    tokio::time::timeout(Duration::from_secs(2), async {
        while server.state.live_session_count().await != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("A should unregister after disconnect");

    b.send_frame(&ClientFrame::Finalize).await;
    assert_eq!(
        b.recv_frame().await,
        ServerFrame::Final {
            text: "final-0".into()
        }
    );
    assert!(matches!(b.recv_raw().await, WsMessage::Close(_)));

    server.shutdown().await;
}