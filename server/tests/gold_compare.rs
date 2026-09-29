//! Gold transcript compare harness (Phase 6.2): one correctness clip.
//!
//! Host-only tests exercise manifest / gold / normalize without a model.
//! The ignored ORT test runs the full clip through `OrtParakeetEngine` and
//! compares Final text to the committed gold.
//!
//! ```text
//! RMLK_NEMOTRON_MODEL_DIR=... RMLK_ASR_CORPUS_DIR=... \
//!   cargo test -p rmlk-server --features ort --test gold_compare -- --ignored --nocapture
//! ```
//!
//! See `server/docs/corpus.md`.

#[path = "common/corpus.rs"]
mod corpus;

use corpus::{
    assert_gold_match, load_gold, load_manifest, normalize_transcript, sha256_hex, verify_wav,
    CORPUS_DIR_ENV,
};

#[cfg(feature = "ort")]
use corpus::resolve_wav;

#[test]
fn manifest_lists_utt001_with_stable_checksum() {
    let manifest = load_manifest();
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.corpus, "correctness");
    assert_eq!(manifest.sample_rate_hz, 16_000);
    assert_eq!(manifest.channels, 1);
    assert_eq!(manifest.pcm, "s16le");
    assert_eq!(manifest.chunk_samples, 8960);

    let clip = manifest.clip("utt001");
    assert_eq!(clip.wav, "utt001.wav");
    assert_eq!(clip.gold, "utt001.gold");
    assert_eq!(clip.bytes, 438_158);
    assert_eq!(
        clip.sha256,
        "0cef37e8097c8f933db56ea00be208926a284e2294af8bcb9cc5c065b4a3113f"
    );
    assert_eq!(clip.sha256.len(), 64);
}

#[test]
fn utt001_gold_file_is_non_empty_and_normalizes() {
    let manifest = load_manifest();
    let clip = manifest.clip("utt001");
    let gold = load_gold(&manifest, clip);
    assert!(
        gold.split_whitespace().count() >= 20,
        "utt001 gold should be a full sentence; got {gold:?}"
    );
    let norm = normalize_transcript(&gold);
    assert!(norm.contains("slushy country roads"));
    assert!(norm.contains("draughtty schoolrooms"));
}

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
    // Apostrophes are treated as punctuation, so contractions split.
    assert_eq!(normalize_transcript("he'll"), "he ll");
    assert_eq!(normalize_transcript("already normalized text"), "already normalized text");
    assert_eq!(normalize_transcript("Café"), "café");
}

#[test]
fn assert_gold_match_accepts_punct_and_case_differences() {
    let gold = "Hello, world.";
    assert_gold_match("hello world", gold);
    assert_gold_match("HELLO WORLD!", gold);
}

#[test]
fn assert_gold_match_rejects_word_errors() {
    let gold = "hello world";
    let err = std::panic::catch_unwind(|| assert_gold_match("hello there", gold));
    assert!(err.is_err());
}

#[test]
fn verify_wav_accepts_matching_utt001_when_present() {
    let manifest = load_manifest();
    let clip = manifest.clip("utt001");

    let wav = local_utt001_wav(&manifest, clip);
    let Some(wav) = wav else {
        // Host CI has no audio; mismatch path below still runs.
        eprintln!("skipping verify_wav accept check: no local utt001 WAV");
        return;
    };
    verify_wav(&wav, clip).expect("utt001 WAV should match manifest checksum");
    assert_eq!(sha256_hex(&std::fs::read(&wav).unwrap()), clip.sha256);
}

fn local_utt001_wav(
    manifest: &corpus::Manifest,
    clip: &corpus::Clip,
) -> Option<std::path::PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(dir) = std::env::var(CORPUS_DIR_ENV) {
        candidates.push(std::path::PathBuf::from(dir).join(&clip.wav));
    }
    candidates.push(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join(".cache/rmlk/asr")
            .join(&manifest.corpus)
            .join(&manifest.corpus_edition)
            .join(&clip.wav),
    );
    candidates.into_iter().find(|p| p.is_file())
}

#[test]
fn verify_wav_rejects_checksum_mismatch() {
    let manifest = load_manifest();
    let clip = manifest.clip("utt001");
    let dir = tempfile::tempdir().unwrap();
    let bogus = dir.path().join(&clip.wav);
    // Same size as utt001 so we hit the hash check, not the size check.
    let mut bytes = vec![0u8; clip.bytes as usize];
    bytes[0] = 1;
    std::fs::write(&bogus, &bytes).unwrap();

    let err = verify_wav(&bogus, clip).expect_err("expected checksum mismatch");
    assert!(
        err.contains("sha256 mismatch"),
        "unexpected error: {err}"
    );
}

#[cfg(feature = "ort")]
mod ort_gold {
    use super::*;
    use std::path::Path;

    use rmlk_server::engine::{Engine, EngineEvent, OrtParakeetEngine, CHUNK_SAMPLES, MODEL_DIR_ENV};
    use rmlk_server::session::StreamState;

    fn load_wav_pcm16(path: &Path) -> Vec<i16> {
        let mut reader = hound::WavReader::open(path).unwrap_or_else(|err| {
            panic!("failed to open WAV {}: {err}", path.display())
        });
        let spec = reader.spec();
        assert_eq!(spec.channels, 1, "fixture must be mono");
        assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        assert_eq!(spec.bits_per_sample, 16);
        reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|err| panic!("failed to read WAV samples: {err}"))
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

    #[test]
    #[ignore = "requires model + utt001 WAV; set RMLK_NEMOTRON_MODEL_DIR and RMLK_ASR_CORPUS_DIR (or RMLK_ASR_UTT_UTT001_WAV)"]
    fn utt001_final_matches_gold() {
        let model_dir = std::env::var(MODEL_DIR_ENV).unwrap_or_else(|_| {
            panic!("{MODEL_DIR_ENV} must point at a Nemotron ONNX directory")
        });
        let manifest = load_manifest();
        let clip = manifest.clip("utt001");
        let wav_path = resolve_wav(&manifest, clip);
        let gold = load_gold(&manifest, clip);

        let mut engine = OrtParakeetEngine::load(&model_dir).expect("load model");
        let pcm = load_wav_pcm16(&wav_path);
        assert!(
            pcm.len() > CHUNK_SAMPLES,
            "utt001 must span more than one chunk; got {} samples",
            pcm.len()
        );

        let mut state = StreamState::default();
        engine.open_stream(&mut state).unwrap();
        for (i, chunk) in pcm_chunks(&pcm, CHUNK_SAMPLES).iter().enumerate() {
            let events = engine.push_audio(&mut state, chunk).unwrap_or_else(|err| {
                panic!("push_audio failed on chunk {i}: {err}")
            });
            for event in events {
                assert!(
                    matches!(event, EngineEvent::Partial { .. }),
                    "unexpected event on chunk {i}: {event:?}"
                );
            }
        }
        let finals = engine.finalize(&mut state).expect("finalize");
        assert_eq!(finals.len(), 1);
        let EngineEvent::Final { text } = &finals[0] else {
            panic!("expected Final, got {:?}", finals[0]);
        };
        println!("utt001 final: {text}");
        assert_gold_match(text, &gold);
    }
}
