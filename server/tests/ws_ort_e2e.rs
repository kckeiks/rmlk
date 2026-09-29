//! Ignored e2e: WebSocket client → server with real OrtParakeetEngine.
//!
//! Run with:
//! ```text
//! RMLK_NEMOTRON_MODEL_DIR=... RMLK_ASR_FIXTURE_WAV=... \
//!   cargo test -p rmlk-server --features ort --test ws_ort_e2e -- --ignored --nocapture
//! ```

#![cfg(feature = "ort")]

mod common;

use std::path::Path;

use rmlk_server::engine::{CHUNK_SAMPLES, MODEL_DIR_ENV};
use rmlk_server::http::app_state_from_config;
use rmlk_server::protocol::{ClientFrame, ServerFrame};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use common::TestServer;

const FIXTURE_WAV_ENV: &str = "RMLK_ASR_FIXTURE_WAV";

fn load_wav_pcm16(path: &Path) -> Vec<i16> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|err| {
        panic!("failed to open fixture WAV {}: {err}", path.display())
    });
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "fixture must be mono");
    assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    assert_eq!(spec.bits_per_sample, 16);
    reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|err| panic!("failed to read fixture WAV samples: {err}"))
}

fn pcm_chunks(pcm: &[i16], chunk_samples: usize) -> Vec<Vec<i16>> {
    pcm.chunks(chunk_samples)
        .map(|chunk| {
            let mut padded = chunk.to_vec();
            if padded.len() < chunk_samples {
                padded.resize(chunk_samples, 0);
            }
            padded
        })
        .collect()
}

#[tokio::test]
#[ignore = "requires model + WAV; set RMLK_NEMOTRON_MODEL_DIR and RMLK_ASR_FIXTURE_WAV"]
async fn ws_client_one_real_call_returns_final_transcript() {
    let model_dir = std::env::var(MODEL_DIR_ENV).unwrap_or_else(|_| {
        panic!("{MODEL_DIR_ENV} must point at a Nemotron ONNX directory")
    });
    let wav_path = std::env::var(FIXTURE_WAV_ENV).unwrap_or_else(|_| {
        panic!("{FIXTURE_WAV_ENV} must point at a 16 kHz mono PCM16 WAV")
    });

    let state = app_state_from_config("ort", Some(Path::new(&model_dir)))
        .expect("load ort app state");
    assert_eq!(state.engine_name(), "ort");

    let server = TestServer::spawn(state).await;
    let mut client = server.connect().await;
    let session_id = client.open_session().await;
    println!("opened session_id={session_id}");

    let pcm = load_wav_pcm16(Path::new(&wav_path));
    assert!(
        pcm.len() > CHUNK_SAMPLES,
        "fixture must span more than one chunk; got {} samples",
        pcm.len()
    );
    let chunks = pcm_chunks(&pcm, CHUNK_SAMPLES);

    for chunk in &chunks {
        client
            .send_frame(&ClientFrame::Audio {
                pcm16: chunk.clone(),
            })
            .await;
    }
    client.send_frame(&ClientFrame::Finalize).await;

    let mut saw_partial = false;
    let final_text = loop {
        match client.recv_frame().await {
            ServerFrame::Partial { text } => {
                println!("partial: {text}");
                saw_partial = true;
            }
            ServerFrame::Final { text } => break text,
            other => panic!("unexpected server frame: {other:?}"),
        }
    };
    println!("final: {final_text}");
    assert!(
        !final_text.trim().is_empty(),
        "expected non-empty Final transcript"
    );
    assert!(
        final_text.split_whitespace().count() >= 2,
        "expected ≥2 words over WS; got {final_text:?}"
    );
    // Real speech usually emits at least one Partial before Final.
    let _ = saw_partial;

    assert!(matches!(client.recv_raw().await, WsMessage::Close(_)));
    server.shutdown().await;
}
