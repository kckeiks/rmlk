//! ASR e2e: normalize helpers + ignored real-engine WebSocket gates.
//!
//! Always-on tests cover [`utils::normalize_transcript`] edge cases.
//! Ignored ORT tests stream `e2e.json` clips over WS.
//!
//! ```text
//! # pack WAVs + rewrite e2e manifest (once):
//! python3 server/scripts/pack_librispeech.py --config clean --split test \
//!   --id 6930-75918-0000,6930-75918-0001,6930-75918-0002
//!
//! RMLK_NEMOTRON_MODEL_DIR=... \
//!   cargo test -p rmlk-server --features ort --test e2e -- --ignored --nocapture
//! ```
//!
//! See `server/docs/tests.md`.

mod utils;

use utils::normalize_transcript;

#[test]
fn normalize_lowercases_and_strips_punct() {
    assert_eq!(normalize_transcript("Hello, World!"), "hello world");
    assert_eq!(
        normalize_transcript("  draughtty   schoolrooms. "),
        "draughtty schoolrooms"
    );
}

#[test]
fn normalize_transcript_edge_cases() {
    assert_eq!(normalize_transcript(""), "");
    assert_eq!(normalize_transcript("   "), "");
    assert_eq!(normalize_transcript("!!!"), "");
    assert_eq!(normalize_transcript("...hello..."), "hello");
    assert_eq!(normalize_transcript("hello\t\nworld"), "hello world");
    assert_eq!(normalize_transcript("HELLO"), "hello");
    assert_eq!(normalize_transcript("abc123"), "abc123");
    assert_eq!(normalize_transcript("a-b"), "a b");
    assert_eq!(normalize_transcript("he'll"), "he ll");
    assert_eq!(
        normalize_transcript("already normalized text"),
        "already normalized text"
    );
    assert_eq!(normalize_transcript("Café"), "café");
}

#[cfg(feature = "ort")]
mod ort_ws {
    use super::utils::{
        assert_normalized_equal, load_manifest, load_wav_pcm16, pcm_chunks, resolve_wav, TestServer,
    };

    use std::path::Path;

    use rmlk_server::engine::{CHUNK_SAMPLES, MODEL_DIR_ENV};
    use rmlk_server::http::app_state_from_config;
    use rmlk_server::protocol::{ClientFrame, ServerFrame};
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    #[tokio::test]
    #[ignore = "requires model + packed WAVs; set RMLK_NEMOTRON_MODEL_DIR (see tests.md)"]
    async fn e2e_clips_match_reference_over_ws() {
        let model_dir = std::env::var(MODEL_DIR_ENV)
            .unwrap_or_else(|_| panic!("{MODEL_DIR_ENV} must point at a Nemotron ONNX directory"));
        let manifest = load_manifest();
        let state =
            app_state_from_config("ort", Some(Path::new(&model_dir))).expect("load ort app state");
        assert_eq!(state.engine_name(), "ort");

        let server = TestServer::spawn(state).await;

        for clip in &manifest.clips {
            let wav_path = resolve_wav(&manifest, clip);
            let pcm = load_wav_pcm16(&wav_path);
            assert!(
                !pcm.is_empty(),
                "{}: empty WAV {}",
                clip.id,
                wav_path.display()
            );
            let chunks = pcm_chunks(&pcm, CHUNK_SAMPLES);

            let mut client = server.connect().await;
            let session_id = client.open_session().await;
            println!("{} session_id={session_id}", clip.id);

            for chunk in &chunks {
                client
                    .send_frame(&ClientFrame::Audio {
                        pcm16: chunk.clone(),
                    })
                    .await;
            }
            client.send_frame(&ClientFrame::Finalize).await;

            let final_text = loop {
                match client.recv_frame().await {
                    ServerFrame::Partial { text } => println!("{} partial: {text}", clip.id),
                    ServerFrame::Final { text } => break text,
                    other => panic!("{}: unexpected server frame: {other:?}", clip.id),
                }
            };
            println!("{} final: {final_text}", clip.id);
            assert_normalized_equal(&final_text, &clip.reference);

            assert!(matches!(client.recv_raw().await, WsMessage::Close(_)));
        }

        server.shutdown().await;
    }

    /// Cancel after streaming only part of a clip: Close, no Final, registry empty.
    #[tokio::test]
    #[ignore = "requires model + packed WAVs; set RMLK_NEMOTRON_MODEL_DIR (see tests.md)"]
    async fn cancel_mid_utterance_over_ws() {
        let model_dir = std::env::var(MODEL_DIR_ENV)
            .unwrap_or_else(|_| panic!("{MODEL_DIR_ENV} must point at a Nemotron ONNX directory"));
        let manifest = load_manifest();
        let clip = manifest
            .clips
            .first()
            .expect("e2e manifest must list at least one clip");
        let state =
            app_state_from_config("ort", Some(Path::new(&model_dir))).expect("load ort app state");
        assert_eq!(state.engine_name(), "ort");

        let server = TestServer::spawn(state).await;
        let wav_path = resolve_wav(&manifest, clip);
        let pcm = load_wav_pcm16(&wav_path);
        let chunks = pcm_chunks(&pcm, CHUNK_SAMPLES);
        assert!(
            chunks.len() >= 2,
            "{}: need at least 2 chunks to cancel mid-utterance, got {}",
            clip.id,
            chunks.len()
        );
        let mid = chunks.len() / 2;

        let mut client = server.connect().await;
        let session_id = client.open_session().await;
        println!(
            "{} session_id={session_id} cancel after {mid}/{} chunks",
            clip.id,
            chunks.len()
        );
        assert_eq!(server.state.live_session_count().await, 1);

        for chunk in &chunks[..mid] {
            client
                .send_frame(&ClientFrame::Audio {
                    pcm16: chunk.clone(),
                })
                .await;
        }
        client.send_frame(&ClientFrame::Cancel).await;

        // Drain any Partial frames still buffered from the last Audio, then Close.
        // A Final must never appear after Cancel.
        loop {
            match client.recv_raw().await {
                WsMessage::Binary(bytes) => match ServerFrame::decode(&bytes) {
                    Ok(ServerFrame::Partial { text }) => {
                        println!("{} partial (pre-cancel drain): {text}", clip.id);
                    }
                    Ok(ServerFrame::Final { text }) => {
                        panic!("{}: unexpected Final after Cancel: {text}", clip.id);
                    }
                    Ok(other) => panic!("{}: unexpected server frame: {other:?}", clip.id),
                    Err(err) => panic!("{}: decode error: {err}", clip.id),
                },
                WsMessage::Close(_) => break,
                other => panic!("{}: unexpected ws message: {other:?}", clip.id),
            }
        }

        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while server.state.live_session_count().await != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("session should unregister after mid-utterance cancel");

        server.shutdown().await;
    }

    /// Finalize then pipeline extra Audio: one Final, Close, no second utterance.
    #[tokio::test]
    #[ignore = "requires model + packed WAVs; set RMLK_NEMOTRON_MODEL_DIR (see tests.md)"]
    async fn finalize_then_audio_rejected_over_ws() {
        let model_dir = std::env::var(MODEL_DIR_ENV)
            .unwrap_or_else(|_| panic!("{MODEL_DIR_ENV} must point at a Nemotron ONNX directory"));
        let manifest = load_manifest();
        let clip = manifest
            .clips
            .first()
            .expect("e2e manifest must list at least one clip");
        let state =
            app_state_from_config("ort", Some(Path::new(&model_dir))).expect("load ort app state");
        assert_eq!(state.engine_name(), "ort");

        let server = TestServer::spawn(state).await;
        let wav_path = resolve_wav(&manifest, clip);
        let pcm = load_wav_pcm16(&wav_path);
        let chunks = pcm_chunks(&pcm, CHUNK_SAMPLES);
        assert!(
            !chunks.is_empty(),
            "{}: need at least one chunk",
            clip.id
        );

        let mut client = server.connect().await;
        let session_id = client.open_session().await;
        println!("{} session_id={session_id} finalize then late audio", clip.id);
        assert_eq!(server.state.live_session_count().await, 1);

        for chunk in &chunks {
            client
                .send_frame(&ClientFrame::Audio {
                    pcm16: chunk.clone(),
                })
                .await;
        }

        // Pipeline Finalize with an extra Audio chunk. Trailing audio must not
        // start another utterance (no second Final / post-Final Partial).
        client.send_frame(&ClientFrame::Finalize).await;
        client
            .send_frame(&ClientFrame::Audio {
                pcm16: vec![0; CHUNK_SAMPLES],
            })
            .await;

        let mut saw_final = false;
        loop {
            match client.recv_raw().await {
                WsMessage::Binary(bytes) => match ServerFrame::decode(&bytes) {
                    Ok(ServerFrame::Partial { text }) => {
                        assert!(
                            !saw_final,
                            "{}: Partial after Final (late audio was applied): {text}",
                            clip.id
                        );
                        println!("{} partial: {text}", clip.id);
                    }
                    Ok(ServerFrame::Final { text }) => {
                        assert!(!saw_final, "{}: unexpected second Final: {text}", clip.id);
                        println!("{} final: {text}", clip.id);
                        saw_final = true;
                    }
                    Ok(other) => panic!("{}: unexpected server frame: {other:?}", clip.id),
                    Err(err) => panic!("{}: decode error: {err}", clip.id),
                },
                WsMessage::Close(_) => break,
                other => panic!("{}: unexpected ws message: {other:?}", clip.id),
            }
        }
        assert!(saw_final, "{}: expected Final before Close", clip.id);

        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while server.state.live_session_count().await != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("session should unregister after finalize (late audio ignored)");

        server.shutdown().await;
    }
}
