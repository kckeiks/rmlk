//! Inference backend trait.
//!
//! Protocol and sessions talk only to this trait so backend types stay out of
//! the wire API.

#[cfg(feature = "ort")]
mod parakeet;
#[cfg(feature = "ort")]
pub use parakeet::{OrtParakeetEngine, CHUNK_SAMPLES, MODEL_DIR_ENV};

use thiserror::Error;

use crate::protocol::ServerFrame;
use crate::session::StreamState;

/// Transcript events produced by an engine step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    Partial { text: String },
    Final { text: String },
}

impl EngineEvent {
    /// Map to a wire transcript frame (`Partial` / `Final`).
    pub fn into_server_frame(self) -> ServerFrame {
        match self {
            Self::Partial { text } => ServerFrame::Partial { text },
            Self::Final { text } => ServerFrame::Final { text },
        }
    }
}

/// Inference backend failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EngineError {
    #[error("{0}")]
    Failed(String),
}

/// Streaming inference backend: audio chunks in, transcript events out.
///
/// Lifecycle per call: `open_stream`, then zero or more `push_audio` calls,
/// then either `finalize` or `cancel`.
///
/// [`CallState`] is monomorphized into [`StreamState`]; there is no type
/// erasure.
pub trait Engine: Send {
    /// Per-call caches / tokens / transcript owned by the connection task.
    type CallState: Send;

    /// Prepare per-call inference state when a session opens.
    fn open_stream(&mut self, state: &mut StreamState<Self::CallState>) -> Result<(), EngineError>;

    /// Feed mono PCM16 samples. Emits at most one partial transcript.
    fn push_audio(
        &mut self,
        state: &mut StreamState<Self::CallState>,
        pcm16: &[i16],
    ) -> Result<Option<EngineEvent>, EngineError>;

    /// End of audio for this stream. Emits the final transcript event.
    fn finalize(
        &mut self,
        state: &mut StreamState<Self::CallState>,
    ) -> Result<EngineEvent, EngineError>;

    /// Drop per-call inference state without emitting a final transcript.
    fn cancel(&mut self, state: &mut StreamState<Self::CallState>) -> Result<(), EngineError>;
}

/// Deterministic engine for tests: text derived from chunk count only.
#[derive(Debug, Default)]
pub struct MockEngine;

impl MockEngine {
    /// Create a mock engine.
    pub fn new() -> Self {
        Self
    }
}

impl Engine for MockEngine {
    type CallState = ();

    fn open_stream(
        &mut self,
        _state: &mut StreamState<Self::CallState>,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn push_audio(
        &mut self,
        state: &mut StreamState<Self::CallState>,
        _pcm16: &[i16],
    ) -> Result<Option<EngineEvent>, EngineError> {
        let n = state.record_chunk();
        Ok(Some(EngineEvent::Partial {
            text: format!("partial-{n}"),
        }))
    }

    fn finalize(
        &mut self,
        state: &mut StreamState<Self::CallState>,
    ) -> Result<EngineEvent, EngineError> {
        Ok(EngineEvent::Final {
            text: format!("final-{}", state.chunks_pushed()),
        })
    }

    fn cancel(&mut self, _state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine, EngineError, EngineEvent, MockEngine};
    use crate::protocol::ServerFrame;
    use crate::session::{SessionId, StreamState};

    struct NoopEngine;

    impl Engine for NoopEngine {
        type CallState = ();

        fn open_stream(
            &mut self,
            _state: &mut StreamState<Self::CallState>,
        ) -> Result<(), EngineError> {
            Ok(())
        }

        fn push_audio(
            &mut self,
            _state: &mut StreamState<Self::CallState>,
            _pcm16: &[i16],
        ) -> Result<Option<EngineEvent>, EngineError> {
            Ok(None)
        }

        fn finalize(
            &mut self,
            _state: &mut StreamState<Self::CallState>,
        ) -> Result<EngineEvent, EngineError> {
            Ok(EngineEvent::Final {
                text: String::new(),
            })
        }

        fn cancel(&mut self, _state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct SpyEngine {
        opens: u32,
        cancels: u32,
    }

    impl Engine for SpyEngine {
        type CallState = ();

        fn open_stream(
            &mut self,
            _state: &mut StreamState<Self::CallState>,
        ) -> Result<(), EngineError> {
            self.opens += 1;
            Ok(())
        }

        fn push_audio(
            &mut self,
            state: &mut StreamState<Self::CallState>,
            pcm16: &[i16],
        ) -> Result<Option<EngineEvent>, EngineError> {
            MockEngine.push_audio(state, pcm16)
        }

        fn finalize(
            &mut self,
            state: &mut StreamState<Self::CallState>,
        ) -> Result<EngineEvent, EngineError> {
            MockEngine.finalize(state)
        }

        fn cancel(&mut self, _state: &mut StreamState<Self::CallState>) -> Result<(), EngineError> {
            self.cancels += 1;
            Ok(())
        }
    }

    #[test]
    fn engine_event_maps_to_protocol_partial_and_final() {
        assert_eq!(
            EngineEvent::Partial {
                text: "hello".into()
            }
            .into_server_frame(),
            ServerFrame::Partial {
                text: "hello".into()
            }
        );
        assert_eq!(
            EngineEvent::Final {
                text: "hello world".into()
            }
            .into_server_frame(),
            ServerFrame::Final {
                text: "hello world".into()
            }
        );
        assert_eq!(
            EngineEvent::Partial {
                text: String::new()
            }
            .into_server_frame(),
            ServerFrame::Partial {
                text: String::new()
            }
        );
    }

    #[test]
    fn engine_trait_object_smoke() {
        let mut engine: Box<dyn Engine<CallState = ()>> = Box::new(NoopEngine);
        let mut state = StreamState::new(SessionId::from_raw(1), 16);
        engine.open_stream(&mut state).unwrap();
        assert!(engine.push_audio(&mut state, &[]).unwrap().is_none());
        assert_eq!(
            engine.finalize(&mut state).unwrap(),
            EngineEvent::Final {
                text: String::new()
            }
        );
    }

    #[test]
    fn open_stream_and_cancel_lifecycle() {
        let mut engine = SpyEngine::default();
        let mut state = StreamState::new(SessionId::from_raw(1), 16);

        engine.open_stream(&mut state).unwrap();
        engine.push_audio(&mut state, &[0]).unwrap();
        engine.cancel(&mut state).unwrap();

        assert_eq!(engine.opens, 1);
        assert_eq!(engine.cancels, 1);
    }

    #[test]
    fn mock_partials_follow_push_count_not_sample_len() {
        let mut engine = MockEngine::new();
        let mut state = StreamState::default();

        assert_eq!(
            engine.push_audio(&mut state, &[0; 8]).unwrap(),
            Some(EngineEvent::Partial {
                text: "partial-1".into()
            })
        );
        assert_eq!(
            engine.push_audio(&mut state, &[0]).unwrap(),
            Some(EngineEvent::Partial {
                text: "partial-2".into()
            })
        );
        assert_eq!(state.chunks_pushed(), 2);
    }

    #[test]
    fn mock_final_uses_chunk_count() {
        let mut engine = MockEngine::new();
        let mut state = StreamState::default();

        engine.push_audio(&mut state, &[]).unwrap();
        engine.push_audio(&mut state, &[]).unwrap();
        assert_eq!(
            engine.finalize(&mut state).unwrap(),
            EngineEvent::Final {
                text: "final-2".into()
            }
        );
    }

    #[test]
    fn mock_final_with_zero_chunks() {
        let mut engine = MockEngine::new();
        let mut state = StreamState::default();
        assert_eq!(
            engine.finalize(&mut state).unwrap(),
            EngineEvent::Final {
                text: "final-0".into()
            }
        );
    }
}
