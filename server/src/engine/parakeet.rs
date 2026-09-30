//! Nemotron / Parakeet streaming engine backed by parakeet-rs + ORT.

use std::path::{Path, PathBuf};

use parakeet_rs::{Nemotron, NemotronHandle};

use super::{Engine, EngineError, EngineEvent};
use crate::session::StreamState;

/// Files required beside the ONNX graph for [`OrtParakeetEngine::load`].
const REQUIRED_FILES: &[&str] = &["encoder.onnx", "decoder_joint.onnx", "tokenizer.model"];

/// Nemotron streaming step size: 560 ms mono @ 16 kHz.
pub const CHUNK_SAMPLES: usize = 8960;

/// Env var for the ignored real-model load / inference tests.
pub const MODEL_DIR_ENV: &str = "RMLK_NEMOTRON_MODEL_DIR";

/// ORT-backed Nemotron streaming engine (shared model handle).
pub struct OrtParakeetEngine {
    handle: NemotronHandle,
    model_dir: PathBuf,
}

impl OrtParakeetEngine {
    /// Load ONNX + tokenizer from `model_dir`.
    ///
    /// Expected layout (Nemotron English or multilingual 3.5 export):
    /// `encoder.onnx` (+ optional `encoder.onnx.data`), `decoder_joint.onnx`,
    /// `tokenizer.model`.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, EngineError> {
        let model_dir = model_dir.as_ref();
        if !model_dir.is_dir() {
            return Err(EngineError::Failed(format!(
                "model directory not found: {}",
                model_dir.display()
            )));
        }
        for name in REQUIRED_FILES {
            let path = model_dir.join(name);
            if !path.is_file() {
                return Err(EngineError::Failed(format!(
                    "missing required model file: {}",
                    path.display()
                )));
            }
        }

        let handle = NemotronHandle::load(model_dir, None).map_err(|err| {
            EngineError::Failed(format!(
                "failed to load Nemotron model from {}: {err}",
                model_dir.display()
            ))
        })?;

        Ok(Self {
            handle,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// Directory this engine was loaded from.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Shared Nemotron handle (for later per-call `Nemotron::from_shared`).
    pub fn handle(&self) -> &NemotronHandle {
        &self.handle
    }
}

fn pcm16_to_f32(pcm16: &[i16]) -> Vec<f32> {
    pcm16.iter().map(|&s| s as f32 / 32768.0).collect()
}

/// Split PCM into fixed-size streaming steps, zero-padding the last chunk.
#[cfg(test)]
fn pcm_chunks(pcm: &[i16], chunk_samples: usize) -> Vec<Vec<i16>> {
    if pcm.is_empty() || chunk_samples == 0 {
        return Vec::new();
    }
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

fn call_mut(state: &mut StreamState<Nemotron>) -> Result<&mut Nemotron, EngineError> {
    state
        .engine_call_mut()
        .ok_or_else(|| EngineError::Failed("stream not open; call open_stream first".into()))
}

/// Silent chunks fed at finalize to drain the streaming decoder
/// (same pattern as parakeet-rs `examples/streaming.rs`).
const FLUSH_SILENCE_CHUNKS: usize = 3;

impl Engine for OrtParakeetEngine {
    type CallState = Nemotron;

    fn open_stream(&mut self, state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
        // Per-call caches / decoder state / transcript live inside `Nemotron`.
        state.set_engine_call(Nemotron::from_shared(&self.handle));
        Ok(())
    }

    fn push_audio(
        &mut self,
        state: &mut StreamState<Self::CallState>,
        pcm16: &[i16],
    ) -> Result<Option<EngineEvent>, EngineError> {
        let audio = pcm16_to_f32(pcm16);
        let text = {
            let call = call_mut(state)?;
            call.transcribe_chunk(&audio).map_err(|err| {
                EngineError::Failed(format!("Nemotron transcribe_chunk failed: {err}"))
            })?;
            call.get_transcript()
        };
        state.record_chunk();
        if text.is_empty() {
            Ok(None)
        } else {
            Ok(Some(EngineEvent::Partial { text }))
        }
    }

    fn finalize(
        &mut self,
        state: &mut StreamState<Self::CallState>,
    ) -> Result<EngineEvent, EngineError> {
        if state.has_engine_call() {
            let silence = vec![0.0f32; CHUNK_SAMPLES];
            let call = call_mut(state)?;
            for _ in 0..FLUSH_SILENCE_CHUNKS {
                call.transcribe_chunk(&silence).map_err(|err| {
                    EngineError::Failed(format!("Nemotron flush chunk failed: {err}"))
                })?;
            }
        }
        let text = state
            .engine_call()
            .map(|call| call.get_transcript())
            .unwrap_or_default();
        state.clear_engine_call();
        Ok(EngineEvent::Final { text })
    }

    fn cancel(&mut self, state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
        state.clear_engine_call();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pcm16_to_f32, pcm_chunks, OrtParakeetEngine, CHUNK_SAMPLES, MODEL_DIR_ENV, REQUIRED_FILES,
    };
    use crate::engine::Engine;
    use crate::session::StreamState;
    use std::path::PathBuf;

    #[test]
    fn load_missing_directory_fails_clearly() {
        let Err(err) = OrtParakeetEngine::load("/no/such/rmlk-nemotron-model-dir") else {
            panic!("expected missing directory error");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("model directory not found"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("rmlk-nemotron-model-dir"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn load_missing_required_file_fails_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let Err(err) = OrtParakeetEngine::load(dir.path()) else {
            panic!("expected missing file error");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("missing required model file"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains(REQUIRED_FILES[0]), "unexpected error: {msg}");
    }

    #[test]
    fn pcm16_to_f32_scales_full_range() {
        let out = pcm16_to_f32(&[0, i16::MAX, i16::MIN]);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - (i16::MAX as f32 / 32768.0)).abs() < f32::EPSILON);
        assert!((out[2] - (i16::MIN as f32 / 32768.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn chunk_samples_is_560ms_at_16khz() {
        assert_eq!(CHUNK_SAMPLES, 8960);
    }

    #[test]
    #[ignore = "requires Nemotron ONNX dir; set RMLK_NEMOTRON_MODEL_DIR"]
    fn load_real_model_dir() {
        let path = std::env::var(MODEL_DIR_ENV)
            .unwrap_or_else(|_| panic!("{MODEL_DIR_ENV} must point at a Nemotron ONNX directory"));
        let path = PathBuf::from(path);
        assert!(
            path.is_dir(),
            "{MODEL_DIR_ENV} is not a directory: {}",
            path.display()
        );
        let engine = OrtParakeetEngine::load(&path).expect("load real model");
        assert_eq!(engine.model_dir(), path.as_path());
        let _ = engine.handle().mode();

        let mut engine = engine;
        let mut state = StreamState::default();
        engine.open_stream(&mut state).unwrap();
        assert!(state.has_engine_call());
        assert!(state.engine_call().is_some());
        engine.cancel(&mut state).unwrap();
        assert!(!state.has_engine_call());
    }

    #[test]
    fn pcm_chunks_pads_last_and_preserves_order() {
        let pcm: Vec<i16> = (0..10).collect();
        let chunks = pcm_chunks(&pcm, 4);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![0, 1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6, 7]);
        assert_eq!(chunks[2], vec![8, 9, 0, 0]);
        assert!(pcm_chunks(&[], 4).is_empty());
    }
}
