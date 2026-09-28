//! Per-call streaming session state and thin live-session registry.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use thiserror::Error;

use crate::engine::{Engine, EngineError, EngineEvent};

/// Default inbound audio queue depth per session.
const DEFAULT_MAILBOX_CAPACITY: usize = 16;

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

/// Per-call streaming state (owned by the connection task on the data path).
///
/// `C` is the engine’s [`Engine::CallState`] (caches / tokens / transcript).
/// Protocol code must not depend on `C`’s fields — only the engine does.
pub struct StreamState<C = ()> {
    session_id: SessionId,
    chunks_pushed: u64,
    drop_counter: Option<Arc<AtomicUsize>>,
    mailbox_capacity: usize,
    mailbox: VecDeque<Vec<i16>>,
    engine_call: Option<C>,
}

impl<C> fmt::Debug for StreamState<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamState")
            .field("session_id", &self.session_id)
            .field("chunks_pushed", &self.chunks_pushed)
            .field("mailbox_len", &self.mailbox.len())
            .field("mailbox_capacity", &self.mailbox_capacity)
            .field("has_engine_call", &self.engine_call.is_some())
            .finish()
    }
}

impl<C> StreamState<C> {
    /// Create state for `session_id` with an empty inbound mailbox.
    pub fn new(session_id: SessionId, mailbox_capacity: usize) -> Self {
        Self {
            session_id,
            chunks_pushed: 0,
            drop_counter: None,
            mailbox_capacity,
            mailbox: VecDeque::new(),
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

    /// Number of audio chunks waiting in the inbound mailbox.
    pub fn mailbox_len(&self) -> usize {
        self.mailbox.len()
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

    /// Enqueue PCM16 samples. Errors with [`SessionError::Busy`] when full.
    pub fn enqueue_audio(&mut self, pcm16: &[i16]) -> Result<(), SessionError> {
        if self.mailbox.len() >= self.mailbox_capacity {
            return Err(SessionError::Busy);
        }
        self.mailbox.push_back(pcm16.to_vec());
        Ok(())
    }

    /// Drain the inbound mailbox through `engine`; return emitted events.
    pub fn process_inbound<E: Engine<CallState = C>>(
        &mut self,
        engine: &mut E,
    ) -> Result<Vec<EngineEvent>, SessionError> {
        let mut events = Vec::new();
        while let Some(chunk) = self.mailbox.pop_front() {
            events.extend(engine.push_audio(self, &chunk)?);
        }
        Ok(events)
    }

    /// Enqueue PCM16 audio and process the mailbox; return emitted events.
    pub fn push_audio<E: Engine<CallState = C>>(
        &mut self,
        engine: &mut E,
        pcm16: &[i16],
    ) -> Result<Vec<EngineEvent>, SessionError> {
        self.enqueue_audio(pcm16)?;
        self.process_inbound(engine)
    }

    /// Flush remaining audio through `engine` and return emitted events.
    pub fn finalize<E: Engine<CallState = C>>(
        mut self,
        engine: &mut E,
    ) -> Result<Vec<EngineEvent>, SessionError> {
        Ok(engine.finalize(&mut self)?)
    }

    /// Cancel through `engine` without a final transcript, then drop state.
    pub fn cancel<E: Engine<CallState = C>>(
        mut self,
        engine: &mut E,
    ) -> Result<(), SessionError> {
        Ok(engine.cancel(&mut self)?)
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
        Self::new(SessionId::default(), DEFAULT_MAILBOX_CAPACITY)
    }
}

/// Session lookup and engine failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SessionError {
    #[error("unknown session {0:?}")]
    Unknown(SessionId),
    #[error("session mailbox full")]
    Busy,
    #[error(transparent)]
    Engine(#[from] EngineError),
}

/// Thin registry of live session ids (no stream state on the data path).
#[derive(Debug)]
pub struct SessionRegistry {
    next_id: u64,
    live: HashSet<SessionId>,
    mailbox_capacity: usize,
}

impl SessionRegistry {
    /// Create an empty registry.
    pub fn new(mailbox_capacity: usize) -> Self {
        Self {
            next_id: 0,
            live: HashSet::new(),
            mailbox_capacity,
        }
    }

    /// Allocate an id, register it as live, and return owned stream state.
    pub fn open<C>(&mut self) -> (SessionId, StreamState<C>) {
        let id = SessionId::from_raw(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.live.insert(id);
        (id, StreamState::new(id, self.mailbox_capacity))
    }

    /// Remove a live id from the registry (caller drops owned [`StreamState`]).
    pub fn unregister(&mut self, id: SessionId) -> Result<(), SessionError> {
        if self.live.remove(&id) {
            Ok(())
        } else {
            Err(SessionError::Unknown(id))
        }
    }

    /// Whether `id` is currently registered.
    pub fn contains(&self, id: SessionId) -> bool {
        self.live.contains(&id)
    }

    /// Number of live sessions.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// Whether there are no live sessions.
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }
}

impl Default for SessionRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_MAILBOX_CAPACITY)
    }
}

/// Registry plus a shared engine handle for tests and the HTTP layer.
#[derive(Debug)]
pub struct Sessions<E> {
    registry: SessionRegistry,
    engine: E,
}

impl<E> Sessions<E> {
    /// Create a session registry backed by `engine`.
    pub fn new(engine: E) -> Self {
        Self::with_mailbox_capacity(engine, DEFAULT_MAILBOX_CAPACITY)
    }

    /// Create a registry with a per-session inbound mailbox capacity.
    pub fn with_mailbox_capacity(engine: E, mailbox_capacity: usize) -> Self {
        Self {
            registry: SessionRegistry::new(mailbox_capacity),
            engine,
        }
    }

    /// Number of live sessions.
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Whether there are no live sessions.
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    /// Whether `id` is registered.
    pub fn contains(&self, id: SessionId) -> bool {
        self.registry.contains(id)
    }

    /// Borrow the engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Borrow the engine mutably.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Unregister a live id (caller drops owned stream state).
    pub fn unregister(&mut self, id: SessionId) -> Result<(), SessionError> {
        self.registry.unregister(id)
    }
}

impl<E: Engine> Sessions<E> {
    /// Open a session: register id and return owned stream state.
    pub fn open(&mut self) -> (SessionId, StreamState<E::CallState>) {
        self.registry.open()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{SessionError, SessionId, SessionRegistry, Sessions, StreamState};
    use crate::engine::{EngineEvent, MockEngine};

    #[test]
    fn open_registers_and_returns_owned_state() {
        let mut registry = SessionRegistry::new(16);
        assert!(registry.is_empty());

        let (id, state) = registry.open::<()>();
        assert_eq!(registry.len(), 1);
        assert!(registry.contains(id));
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
    fn open_allocates_distinct_ids() {
        let mut registry = SessionRegistry::new(16);
        let (a, _) = registry.open::<()>();
        let (b, _) = registry.open::<()>();
        assert_ne!(a, b);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn unregister_unknown_errors() {
        let mut registry = SessionRegistry::new(16);
        let id = SessionId::from_raw(42);
        assert_eq!(
            registry.unregister(id).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn push_uses_owned_state_not_registry() {
        let mut registry = SessionRegistry::new(16);
        let mut engine = MockEngine::new();
        let (id, mut state) = registry.open();

        // Data path: only owned state + engine (registry not borrowed).
        let events = state.push_audio(&mut engine, &[0, 1, 2]).unwrap();
        assert_eq!(
            events,
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        assert_eq!(state.chunks_pushed(), 1);
        assert!(registry.contains(id));
    }

    #[test]
    fn finalize_owned_then_unregister() {
        let mut sessions = Sessions::new(MockEngine::new());
        let (id, mut state) = sessions.open();
        state.push_audio(sessions.engine_mut(), &[0]).unwrap();

        let events = state.finalize(sessions.engine_mut()).unwrap();
        assert_eq!(
            events,
            vec![EngineEvent::Final {
                text: "final-1".into()
            }]
        );
        sessions.unregister(id).unwrap();
        assert!(sessions.is_empty());
        assert_eq!(
            sessions.unregister(id).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn cancel_unregisters_and_drops_state() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut sessions = Sessions::new(MockEngine::new());
        let (id, mut state) = sessions.open();
        state.track_drops(Arc::clone(&drops));
        state.push_audio(sessions.engine_mut(), &[0]).unwrap();

        state.cancel(sessions.engine_mut()).unwrap();
        sessions.unregister(id).unwrap();
        assert!(sessions.is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(
            sessions.unregister(id).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn duplicate_unregister_after_finalize_errors() {
        let mut sessions = Sessions::new(MockEngine::new());
        let (id, state) = sessions.open();
        let _ = state.finalize(sessions.engine_mut()).unwrap();
        sessions.unregister(id).unwrap();
        assert_eq!(
            sessions.unregister(id).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn dropping_owned_state_runs_drop_hooks() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut registry = SessionRegistry::new(16);
            let (_, mut a) = registry.open::<()>();
            let (_, mut b) = registry.open::<()>();
            a.track_drops(Arc::clone(&drops));
            b.track_drops(Arc::clone(&drops));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            let _ = registry;
        }
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn enqueue_busy_when_mailbox_full() {
        let mut state = StreamState::new(SessionId::from_raw(0), 1);
        let mut engine = MockEngine::new();

        state.enqueue_audio(&[1]).unwrap();
        assert_eq!(state.mailbox_len(), 1);
        assert_eq!(state.enqueue_audio(&[2]).unwrap_err(), SessionError::Busy);

        let events = state.process_inbound(&mut engine).unwrap();
        assert_eq!(
            events,
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        state.enqueue_audio(&[3]).unwrap();
    }

    #[test]
    fn two_sessions_interleaved_no_crosstalk() {
        let mut sessions = Sessions::new(MockEngine::new());
        let (a_id, mut a) = sessions.open();
        let (b_id, mut b) = sessions.open();
        assert_ne!(a_id, b_id);

        assert_eq!(
            a.push_audio(sessions.engine_mut(), &[0]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        assert_eq!(
            b.push_audio(sessions.engine_mut(), &[0]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        assert_eq!(
            a.push_audio(sessions.engine_mut(), &[0]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-2".into()
            }]
        );

        assert_eq!(a.chunks_pushed(), 2);
        assert_eq!(b.chunks_pushed(), 1);

        assert_eq!(
            b.finalize(sessions.engine_mut()).unwrap(),
            vec![EngineEvent::Final {
                text: "final-1".into()
            }]
        );
        sessions.unregister(b_id).unwrap();
        assert_eq!(
            a.finalize(sessions.engine_mut()).unwrap(),
            vec![EngineEvent::Final {
                text: "final-2".into()
            }]
        );
        sessions.unregister(a_id).unwrap();
        assert!(sessions.is_empty());
    }
}
