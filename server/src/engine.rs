//! Inference backend trait.
//!
//! Protocol and sessions talk only to this trait so backend types stay out of
//! the wire API.

use thiserror::Error;

use crate::session::StreamState;

/// Transcript events produced by an engine step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    Partial { text: String },
    Final { text: String },
}

/// Inference backend failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EngineError {
    #[error("{0}")]
    Failed(String),
}

/// Streaming inference backend: audio chunks in, transcript events out.
pub trait Engine: Send {
    /// Feed mono PCM16 samples. May emit zero or more partials.
    fn push_audio(
        &mut self,
        state: &mut StreamState,
        pcm16: &[i16],
    ) -> Result<Vec<EngineEvent>, EngineError>;

    /// End of audio for this stream. Emits a final transcript event.
    fn finalize(
        &mut self,
        state: &mut StreamState,
    ) -> Result<Vec<EngineEvent>, EngineError>;
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
    fn push_audio(
        &mut self,
        state: &mut StreamState,
        _pcm16: &[i16],
    ) -> Result<Vec<EngineEvent>, EngineError> {
        let n = state.record_chunk();
        Ok(vec![EngineEvent::Partial {
            text: format!("partial-{n}"),
        }])
    }

    fn finalize(
        &mut self,
        state: &mut StreamState,
    ) -> Result<Vec<EngineEvent>, EngineError> {
        Ok(vec![EngineEvent::Final {
            text: format!("final-{}", state.chunks_pushed()),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::{Engine, EngineError, EngineEvent, MockEngine};
    use crate::session::{SessionId, StreamState};

    struct NoopEngine;

    impl Engine for NoopEngine {
        fn push_audio(
            &mut self,
            _state: &mut StreamState,
            _pcm16: &[i16],
        ) -> Result<Vec<EngineEvent>, EngineError> {
            Ok(Vec::new())
        }

        fn finalize(
            &mut self,
            _state: &mut StreamState,
        ) -> Result<Vec<EngineEvent>, EngineError> {
            Ok(vec![EngineEvent::Final {
                text: String::new(),
            }])
        }
    }

    #[test]
    fn engine_trait_object_smoke() {
        let mut engine: Box<dyn Engine> = Box::new(NoopEngine);
        let mut state = StreamState::new(SessionId::from_raw(1));
        assert!(engine.push_audio(&mut state, &[]).unwrap().is_empty());
        assert_eq!(
            engine.finalize(&mut state).unwrap(),
            vec![EngineEvent::Final {
                text: String::new()
            }]
        );
    }

    #[test]
    fn mock_partials_follow_push_count_not_sample_len() {
        let mut engine = MockEngine::new();
        let mut state = StreamState::default();

        assert_eq!(
            engine.push_audio(&mut state, &[0; 8]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        assert_eq!(
            engine.push_audio(&mut state, &[0]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-2".into()
            }]
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
            vec![EngineEvent::Final {
                text: "final-2".into()
            }]
        );
    }

    #[test]
    fn mock_final_with_zero_chunks() {
        let mut engine = MockEngine::new();
        let mut state = StreamState::default();
        assert_eq!(
            engine.finalize(&mut state).unwrap(),
            vec![EngineEvent::Final {
                text: "final-0".into()
            }]
        );
    }
}
