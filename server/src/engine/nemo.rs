//! In-process NeMo ASR adapter. Only the recognizer crosses threads; streams
//! are constructed, driven and dropped on their worker's dedicated thread.
use std::path::PathBuf;

use nemo_speech::{
    BatchingConfig, Device, RecognitionOptions, Recognizer, RecognizerConfig, Runtime, SampleRate,
    Stream, StreamingConfig,
};

use super::{Engine, EngineError, EngineEvent};
use crate::session::StreamState;

/// Native SDK startup and bounded session admission. One worker per slot lets
/// independent streams overlap inside the SDK's shared-model batch scheduler.
#[derive(Debug, Clone)]
pub struct NemoConfig {
    pub library: PathBuf,
    pub model: PathBuf,
    pub device: Device,
    pub max_sessions: usize,
    pub batch_size: u32,
    pub queue_delay_us: u32,
    pub right_context: i32,
}

impl NemoConfig {
    pub fn new(library: impl Into<PathBuf>, model: impl Into<PathBuf>) -> Self {
        Self {
            library: library.into(),
            model: model.into(),
            device: Device::Gpu(0),
            max_sessions: 8,
            batch_size: 8,
            queue_delay_us: 5_000,
            right_context: 6,
        }
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if !(1..=64).contains(&self.max_sessions) {
            return Err(EngineError::Failed(
                "NeMo max_sessions must be in 1..=64".into(),
            ));
        }
        if self.batch_size == 0 || self.batch_size as usize > self.max_sessions {
            return Err(EngineError::Failed(
                "NeMo batch_size must be in 1..=max_sessions".into(),
            ));
        }
        Ok(())
    }
}

/// Clones share one loaded model; they do not duplicate model weights.
#[derive(Clone)]
pub struct NemoEngine {
    recognizer: Recognizer,
}

impl NemoEngine {
    /// Load the model once before starting workers.
    ///
    /// # Safety
    /// `config.library` and its dependencies must be a trusted native SDK
    /// satisfying `nemo_speech::Runtime::load`'s ABI and threading contract.
    pub unsafe fn load(config: &NemoConfig) -> Result<Self, EngineError> {
        config.validate()?;
        // SAFETY: caller guarantees the native SDK's ABI and trust contract.
        let runtime = unsafe { Runtime::load(&config.library) }.map_err(native_error)?;
        log::info!(
            "NeMo SDK {}: device={:?}, session_slots={}, batch_size={}",
            runtime.version().map_err(native_error)?,
            config.device,
            config.max_sessions,
            config.batch_size
        );
        let mut options = RecognizerConfig::new(&config.model);
        options.device = config.device;
        options.streaming = Some(StreamingConfig {
            rnnt_right_context: config.right_context,
            ..StreamingConfig::default()
        });
        if config.batch_size > 1 {
            options.batching = Some(BatchingConfig {
                max_batch_size: config.batch_size,
                max_queue_delay_us: config.queue_delay_us,
                max_queue_depth: (config.max_sessions * 16) as u32,
                state_arena_slots: config.max_sessions as u32,
            });
        }
        Ok(Self {
            recognizer: runtime.create_recognizer(&options).map_err(native_error)?,
        })
    }
}

pub struct NemoCallState {
    stream: Stream,
    transcript: Transcript,
    emitted: String,
    samples: Vec<f32>,
}

/// Native finals mark utterance endpoints. The wire Final marks connection EOF;
/// retain completed utterances and replace the current interim hypothesis.
#[derive(Default)]
struct Transcript {
    completed: String,
    interim: String,
}

impl Transcript {
    fn update(&mut self, text: &str, is_final: bool) {
        if is_final {
            append(&mut self.completed, text.trim());
            self.interim.clear();
        } else {
            self.interim = text.trim().to_owned();
        }
    }

    fn text(&self) -> String {
        let mut text = self.completed.clone();
        append(&mut text, &self.interim);
        text
    }
}

fn append(target: &mut String, text: &str) {
    if !text.is_empty() {
        if !target.is_empty() {
            target.push(' ');
        }
        target.push_str(text);
    }
}

impl NemoCallState {
    fn drain(&mut self) -> Result<(), EngineError> {
        while let Some(result) = self.stream.next_result().map_err(native_error)? {
            self.transcript
                .update(result.transcript().unwrap_or(""), result.is_final);
        }
        Ok(())
    }
}

impl Engine for NemoEngine {
    type CallState = NemoCallState;

    fn open_stream(&mut self, state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
        let stream = self
            .recognizer
            .create_stream(&RecognitionOptions::default())
            .map_err(native_error)?;
        state.set_engine_call(NemoCallState {
            stream,
            transcript: Transcript::default(),
            emitted: String::new(),
            samples: Vec::new(),
        });
        Ok(())
    }

    fn push_audio(
        &mut self,
        state: &mut StreamState<Self::CallState>,
        pcm16: &[i16],
    ) -> Result<Option<EngineEvent>, EngineError> {
        state.record_chunk();
        let call = state.engine_call_mut().ok_or_else(missing_call)?;
        call.samples.clear();
        call.samples
            .extend(pcm16.iter().map(|&sample| f32::from(sample) / 32768.0));
        call.stream
            .push_audio(&call.samples, SampleRate::HZ_16000)
            .map_err(native_error)?;
        call.drain()?;
        let text = call.transcript.text();
        if text == call.emitted {
            return Ok(None);
        }
        call.emitted = text.clone();
        Ok(Some(EngineEvent::Partial { text }))
    }

    fn finalize(
        &mut self,
        state: &mut StreamState<Self::CallState>,
    ) -> Result<EngineEvent, EngineError> {
        let mut call = state.take_engine_call().ok_or_else(missing_call)?;
        call.stream.finish().map_err(native_error)?;
        call.drain()?;
        Ok(EngineEvent::Final {
            text: call.transcript.text(),
        })
    }

    fn cancel(&mut self, state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
        state.clear_engine_call();
        Ok(())
    }
}

fn missing_call() -> EngineError {
    EngineError::Failed("NeMo stream is not open".into())
}
fn native_error(error: nemo_speech::Error) -> EngineError {
    EngineError::Failed(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_endpoints_replaces_interims_and_preserves_repeated_words() {
        let mut transcript = Transcript::default();
        transcript.update("hel", false);
        transcript.update("hello", false);
        transcript.update("hello", true);
        transcript.update("hello", false);
        assert_eq!(transcript.text(), "hello hello");
        transcript.update("hello world", true);
        assert_eq!(transcript.text(), "hello hello world");
        transcript.update("", true);
        assert_eq!(transcript.text(), "hello hello world");
    }

    #[test]
    fn invalid_admission_is_rejected_before_loading_sdk() {
        let mut config = NemoConfig::new("/missing/sdk", "/missing/model");
        config.max_sessions = 0;
        assert!(config.validate().is_err());
        config.max_sessions = 65;
        assert!(config.validate().is_err());
        config.max_sessions = 4;
        config.batch_size = 5;
        assert!(config.validate().is_err());
        config.batch_size = 0;
        assert!(config.validate().is_err());
        config.batch_size = 4;
        assert!(config.validate().is_ok());
    }
}
