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

#[cfg(test)]
mod tests {
    use super::{Engine, EngineError, EngineEvent};
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
        let mut state = StreamState {
            session_id: SessionId(1),
        };
        assert!(engine.push_audio(&mut state, &[]).unwrap().is_empty());
        assert_eq!(
            engine.finalize(&mut state).unwrap(),
            vec![EngineEvent::Final {
                text: String::new()
            }]
        );
    }
}
