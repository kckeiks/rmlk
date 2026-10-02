//! Real native SDK WebSocket gate; intentionally opt-in, with no downloads.
#![cfg(feature = "nemo")]
mod utils;

use nemo_speech::Device;
use rmlk_server::engine::NemoConfig;
use rmlk_server::http::{app_state_from_nemo_config, AppState, ENGINE_NEMO};
use rmlk_server::protocol::{error_code, ClientFrame, ServerFrame};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;
use utils::{
    assert_normalized_equal, load_manifest, load_wav_pcm16, resolve_wav, TestServer, WsClient,
};

async fn idle(state: &AppState) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while state.live_session_count() != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("native sessions should be released");
    // Live count is decremented just before the stream is dropped. Allow that
    // cleanup to finish before deliberately saturating admission again.
    tokio::time::sleep(Duration::from_millis(20)).await;
}

async fn cancelled(client: &mut WsClient) {
    loop {
        match client.recv_raw().await {
            Message::Close(_) => break,
            Message::Binary(bytes) => assert!(matches!(
                ServerFrame::decode(&bytes).unwrap(),
                ServerFrame::Partial { .. } | ServerFrame::AudioProcessed
            )),
            other => panic!("unexpected cancel response: {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires trusted RMLK_NEMO_LIBRARY, RMLK_NEMO_MODEL, CUDA GPU and cached e2e WAVs; see docs/nemo.md"]
async fn native_websocket_transcription_admission_and_cleanup() {
    let library = std::env::var("RMLK_NEMO_LIBRARY").expect("set RMLK_NEMO_LIBRARY");
    let model = std::env::var("RMLK_NEMO_MODEL").expect("set RMLK_NEMO_MODEL");
    let mut config = NemoConfig::new(library, model);
    config.max_sessions = 4;
    config.batch_size = 4;
    config.device = match std::env::var("RMLK_NEMO_DEVICE").as_deref().unwrap_or("0") {
        "-1" => Device::Cpu,
        index => Device::Gpu(index.parse().expect("GPU index or -1")),
    };
    // SAFETY: this opt-in test's operator supplies the verified pinned SDK.
    let (state, worker) =
        unsafe { app_state_from_nemo_config(&config) }.expect("load native backend");
    assert_eq!(state.engine_name(), ENGINE_NEMO);
    let server = TestServer::spawn(state).await;

    let mut held = Vec::new();
    for _ in 0..4 {
        let mut client = server.connect().await;
        client.open_session().await;
        held.push(client);
    }
    assert_eq!(server.state.live_session_count(), 4);
    let mut excess = server.connect().await;
    excess.send_frame(&ClientFrame::Open).await;
    assert!(matches!(
        excess.recv_frame().await,
        ServerFrame::Error {
            code: error_code::BUSY,
            ..
        }
    ));
    assert!(matches!(excess.recv_raw().await, Message::Close(_)));
    for client in &mut held {
        client.send_frame(&ClientFrame::Cancel).await;
        cancelled(client).await;
    }
    drop(held);
    idle(&server.state).await;

    let manifest = load_manifest();
    let mut clients = Vec::new();
    for _ in 0..4 {
        let mut client = server.connect().await;
        client.open_session().await;
        clients.push(client);
    }
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(4));
    let mut tasks = Vec::new();
    for (index, mut client) in clients.into_iter().enumerate() {
        let clip = &manifest.clips[index % manifest.clips.len()];
        let pcm = load_wav_pcm16(&resolve_wav(&manifest, clip));
        let reference = clip.reference.clone();
        let id = clip.id.clone();
        let chunk_samples = manifest.chunk_samples;
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            let start = tokio::time::Instant::now();
            for (i, chunk) in pcm.chunks(chunk_samples).enumerate() {
                tokio::time::sleep_until(
                    start + Duration::from_secs_f64(i as f64 * chunk_samples as f64 / 16_000.0),
                )
                .await;
                client
                    .send_frame(&ClientFrame::Audio {
                        pcm16: chunk.to_vec(),
                    })
                    .await;
            }
            client.send_frame(&ClientFrame::Finalize).await;
            let text = loop {
                match client.recv_frame().await {
                    ServerFrame::Partial { .. } | ServerFrame::AudioProcessed => {}
                    ServerFrame::Final { text } => break text,
                    other => panic!("{id}: unexpected response: {other:?}"),
                }
            };
            println!("{id}: {text}");
            assert_normalized_equal(&text, &reference);
            assert!(matches!(client.recv_raw().await, Message::Close(_)));
        }));
    }
    tokio::time::timeout(Duration::from_secs(60), async {
        for task in tasks {
            task.await.unwrap();
        }
    })
    .await
    .expect("concurrent transcription must complete");
    idle(&server.state).await;

    // Both cancellation and a dropped connection release native state. A new
    // full-length utterance afterwards proves the recognizer remains usable.
    let clip = &manifest.clips[0];
    let pcm = load_wav_pcm16(&resolve_wav(&manifest, clip));
    let mut cancel = server.connect().await;
    cancel.open_session().await;
    cancel
        .send_frame(&ClientFrame::Audio {
            pcm16: pcm[..manifest.chunk_samples].to_vec(),
        })
        .await;
    cancel.send_frame(&ClientFrame::Cancel).await;
    cancelled(&mut cancel).await;
    let mut dropped = server.connect().await;
    dropped.open_session().await;
    dropped
        .send_frame(&ClientFrame::Audio {
            pcm16: pcm[..manifest.chunk_samples].to_vec(),
        })
        .await;
    dropped.close().await;
    idle(&server.state).await;
    let mut fresh = server.connect().await;
    fresh.open_session().await;
    for chunk in pcm.chunks(manifest.chunk_samples) {
        fresh
            .send_frame(&ClientFrame::Audio {
                pcm16: chunk.to_vec(),
            })
            .await;
    }
    fresh.send_frame(&ClientFrame::Finalize).await;
    loop {
        match fresh.recv_frame().await {
            ServerFrame::Partial { .. } | ServerFrame::AudioProcessed => {}
            ServerFrame::Final { text } => {
                assert_normalized_equal(&text, &clip.reference);
                break;
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }
    assert!(matches!(fresh.recv_raw().await, Message::Close(_)));
    idle(&server.state).await;
    drop(fresh);
    drop(cancel);
    drop(excess);
    server.shutdown().await;
    tokio::time::timeout(Duration::from_secs(10), worker)
        .await
        .expect("workers stop after app state drops")
        .unwrap();
}
