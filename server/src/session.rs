//! Per-call streaming session state.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use thiserror::Error;

use crate::engine::{Engine, EngineError, EngineEvent};

/// Opaque handle for a live streaming call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SessionId(u64);

impl SessionId {
    /// Wrap a raw id (wire / allocator).
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Raw id for the wire protocol (`OpenAck`).
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Per-call streaming state. Connections send empty session metadata; the
/// engine worker creates backend state locally and owns it for the session.
///
/// `C` is the engine's [`Engine::CallState`] (caches, tokens, transcript).
/// Protocol code must not depend on the fields of `C`; only the engine does.
pub struct StreamState<C = ()> {
    session_id: SessionId,
    chunks_pushed: u64,
    drop_counter: Option<Arc<AtomicUsize>>,
    engine_call: Option<C>,
}

impl<C> fmt::Debug for StreamState<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamState")
            .field("session_id", &self.session_id)
            .field("chunks_pushed", &self.chunks_pushed)
            .field("has_engine_call", &self.engine_call.is_some())
            .finish()
    }
}

impl<C> StreamState<C> {
    /// Create fresh state for `session_id`.
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            chunks_pushed: 0,
            drop_counter: None,
            engine_call: None,
        }
    }

    /// Session this state belongs to.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Number of audio chunks pushed so far.
    pub fn chunks_pushed(&self) -> u64 {
        self.chunks_pushed
    }

    /// Record one pushed audio chunk; returns the new count.
    pub fn record_chunk(&mut self) -> u64 {
        self.chunks_pushed += 1;
        self.chunks_pushed
    }

    /// Increment `counter` when this state is dropped (tests / cleanup checks).
    pub fn track_drops(&mut self, counter: Arc<AtomicUsize>) {
        self.drop_counter = Some(counter);
    }

    /// Whether engine-private per-call state is present.
    pub fn has_engine_call(&self) -> bool {
        self.engine_call.is_some()
    }

    /// Store engine-private per-call state (caches / tokens / transcript).
    pub fn set_engine_call(&mut self, value: C) {
        self.engine_call = Some(value);
    }

    /// Borrow engine-private state, if present.
    pub fn engine_call(&self) -> Option<&C> {
        self.engine_call.as_ref()
    }

    /// Mutably borrow engine-private state, if present.
    pub fn engine_call_mut(&mut self) -> Option<&mut C> {
        self.engine_call.as_mut()
    }

    /// Take engine-private state, clearing the slot.
    pub fn take_engine_call(&mut self) -> Option<C> {
        self.engine_call.take()
    }

    /// Drop engine-private per-call state (cancel / reset).
    pub fn clear_engine_call(&mut self) {
        self.engine_call = None;
    }

    /// Finalize through `engine` and return the final transcript event.
    pub fn finalize<E: Engine<CallState = C>>(
        mut self,
        engine: &mut E,
    ) -> Result<EngineEvent, SessionError> {
        Ok(engine.finalize(&mut self)?)
    }

    /// Cancel through `engine` without a final transcript, then drop state.
    pub fn cancel<E: Engine<CallState = C>>(mut self, engine: &mut E) -> Result<(), SessionError> {
        Ok(engine.cancel(&mut self)?)
    }
}

impl StreamState<()> {
    /// Transfer only session metadata into worker-local state. Backend state is
    /// created after this transfer, so thread-bound native handles never travel
    /// through the work channel.
    pub(crate) fn into_worker<C>(mut self) -> StreamState<C> {
        StreamState {
            session_id: self.session_id,
            chunks_pushed: self.chunks_pushed,
            drop_counter: self.drop_counter.take(),
            engine_call: None,
        }
    }
}

impl<C> Drop for StreamState<C> {
    fn drop(&mut self) {
        if let Some(counter) = &self.drop_counter {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl<C> Default for StreamState<C> {
    fn default() -> Self {
        Self::new(SessionId::default())
    }
}

/// Session and engine failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SessionError {
    #[error("session has too much audio waiting for the engine")]
    Busy,
    #[error(transparent)]
    Engine(#[from] EngineError),
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::{SessionId, StreamState};
    use crate::engine::{Engine, EngineEvent, MockEngine};

    #[test]
    fn new_state_starts_empty() {
        let id = SessionId::from_raw(7);
        let state = StreamState::<()>::new(id);
        assert_eq!(state.session_id(), id);
        assert_eq!(state.chunks_pushed(), 0);
        assert!(!state.has_engine_call());
    }

    #[test]
    fn engine_call_state_construct_mutate_reset_without_model() {
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct FakeCallState {
            transcript: String,
            tokens: Vec<u32>,
            cache_ticks: u64,
        }

        let mut state = StreamState::<FakeCallState>::default();
        assert!(!state.has_engine_call());

        state.set_engine_call(FakeCallState {
            transcript: String::new(),
            tokens: Vec::new(),
            cache_ticks: 0,
        });
        assert!(state.has_engine_call());

        {
            let call = state.engine_call_mut().unwrap();
            call.tokens.push(7);
            call.transcript.push_str("hi");
            call.cache_ticks = 3;
        }
        assert_eq!(
            state.engine_call().unwrap(),
            &FakeCallState {
                transcript: "hi".into(),
                tokens: vec![7],
                cache_ticks: 3,
            }
        );

        let taken = state.take_engine_call().unwrap();
        assert_eq!(taken.cache_ticks, 3);
        assert!(!state.has_engine_call());

        state.set_engine_call(FakeCallState {
            transcript: "x".into(),
            tokens: vec![1],
            cache_ticks: 0,
        });
        state.clear_engine_call();
        assert!(!state.has_engine_call());
        assert!(state.engine_call().is_none());
    }

    #[test]
    fn finalize_consumes_state_and_reports_chunk_count() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut engine = MockEngine::new();
        let mut state = StreamState::new(SessionId::from_raw(1));
        state.track_drops(Arc::clone(&drops));
        engine.push_audio(&mut state, &[0]).unwrap();

        let event = state.finalize(&mut engine).unwrap();
        assert_eq!(
            event,
            EngineEvent::Final {
                text: "final-1".into()
            }
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancel_consumes_and_drops_state() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut engine = MockEngine::new();
        let mut state = StreamState::new(SessionId::from_raw(1));
        state.track_drops(Arc::clone(&drops));
        engine.push_audio(&mut state, &[0]).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        state.cancel(&mut engine).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_state_runs_drop_hooks() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut a = StreamState::<()>::new(SessionId::from_raw(1));
            let mut b = StreamState::<()>::new(SessionId::from_raw(2));
            a.track_drops(Arc::clone(&drops));
            b.track_drops(Arc::clone(&drops));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn two_sessions_interleaved_no_crosstalk() {
        let mut engine = MockEngine::new();
        let mut a = StreamState::new(SessionId::from_raw(1));
        let mut b = StreamState::new(SessionId::from_raw(2));

        assert_eq!(
            engine.push_audio(&mut a, &[0]).unwrap(),
            Some(EngineEvent::Partial {
                text: "partial-1".into()
            })
        );
        assert_eq!(
            engine.push_audio(&mut b, &[0]).unwrap(),
            Some(EngineEvent::Partial {
                text: "partial-1".into()
            })
        );
        assert_eq!(
            engine.push_audio(&mut a, &[0]).unwrap(),
            Some(EngineEvent::Partial {
                text: "partial-2".into()
            })
        );

        assert_eq!(a.chunks_pushed(), 2);
        assert_eq!(b.chunks_pushed(), 1);

        assert_eq!(
            b.finalize(&mut engine).unwrap(),
            EngineEvent::Final {
                text: "final-1".into()
            }
        );
        assert_eq!(
            a.finalize(&mut engine).unwrap(),
            EngineEvent::Final {
                text: "final-2".into()
            }
        );
    }
}
