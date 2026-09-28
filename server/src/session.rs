//! Per-call streaming session state.

use std::collections::{HashMap, VecDeque};
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

/// Per-call streaming state.
#[derive(Debug)]
pub struct StreamState {
    session_id: SessionId,
    chunks_pushed: u64,
    drop_counter: Option<Arc<AtomicUsize>>,
    mailbox_capacity: usize,
    mailbox: VecDeque<Vec<i16>>,
}

impl StreamState {
    /// Create state for `session_id` with an empty inbound mailbox.
    pub fn new(session_id: SessionId, mailbox_capacity: usize) -> Self {
        Self {
            session_id,
            chunks_pushed: 0,
            drop_counter: None,
            mailbox_capacity,
            mailbox: VecDeque::new(),
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

    /// Enqueue PCM16 samples. Errors with [`SessionError::Busy`] when full.
    fn try_enqueue(&mut self, pcm16: Vec<i16>) -> Result<(), SessionError> {
        if self.mailbox.len() >= self.mailbox_capacity {
            return Err(SessionError::Busy);
        }
        self.mailbox.push_back(pcm16);
        Ok(())
    }

    /// Pop the next inbound chunk, if any.
    fn pop_inbound(&mut self) -> Option<Vec<i16>> {
        self.mailbox.pop_front()
    }
}

impl Drop for StreamState {
    fn drop(&mut self) {
        if let Some(counter) = &self.drop_counter {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Default for StreamState {
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

/// In-process registry of live sessions.
#[derive(Debug, Default)]
pub struct SessionMap {
    next_id: u64,
    sessions: HashMap<SessionId, StreamState>,
}

impl SessionMap {
    /// Create an empty session map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new session id, insert state with `mailbox_capacity`, return id.
    pub fn create(&mut self, mailbox_capacity: usize) -> SessionId {
        let id = SessionId::from_raw(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.sessions
            .insert(id, StreamState::new(id, mailbox_capacity));
        id
    }

    /// Borrow session state by id.
    pub fn get(&self, id: SessionId) -> Result<&StreamState, SessionError> {
        self.sessions.get(&id).ok_or(SessionError::Unknown(id))
    }

    /// Borrow session state by id, mutably.
    pub fn get_mut(&mut self, id: SessionId) -> Result<&mut StreamState, SessionError> {
        self.sessions.get_mut(&id).ok_or(SessionError::Unknown(id))
    }

    /// Remove and return session state by id.
    pub fn remove(&mut self, id: SessionId) -> Result<StreamState, SessionError> {
        self.sessions
            .remove(&id)
            .ok_or(SessionError::Unknown(id))
    }

    /// Number of live sessions.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the map has no sessions.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

/// In-process session lifecycle over a map and an [`Engine`].
#[derive(Debug)]
pub struct Sessions<E> {
    map: SessionMap,
    engine: E,
    mailbox_capacity: usize,
}

impl<E> Sessions<E> {
    /// Create a session registry backed by `engine`.
    pub fn new(engine: E) -> Self {
        Self::with_mailbox_capacity(engine, DEFAULT_MAILBOX_CAPACITY)
    }

    /// Create a registry with a per-session inbound mailbox capacity.
    pub fn with_mailbox_capacity(engine: E, mailbox_capacity: usize) -> Self {
        Self {
            map: SessionMap::new(),
            engine,
            mailbox_capacity,
        }
    }

    /// Number of live sessions.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether there are no live sessions.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Borrow the engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Borrow the engine mutably.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }
}

impl<E: Engine> Sessions<E> {
    /// Open a session: allocate `StreamState`, register it, return its id.
    pub fn open(&mut self) -> SessionId {
        self.map.create(self.mailbox_capacity)
    }

    /// Borrow session state by id.
    pub fn get(&self, id: SessionId) -> Result<&StreamState, SessionError> {
        self.map.get(id)
    }

    /// Borrow session state by id, mutably.
    pub fn get_mut(&mut self, id: SessionId) -> Result<&mut StreamState, SessionError> {
        self.map.get_mut(id)
    }

    /// Enqueue PCM16 audio without running the engine.
    pub fn enqueue_audio(&mut self, id: SessionId, pcm16: &[i16]) -> Result<(), SessionError> {
        self.map.get_mut(id)?.try_enqueue(pcm16.to_vec())
    }

    /// Drain the inbound mailbox through the engine; return emitted events.
    pub fn process_inbound(&mut self, id: SessionId) -> Result<Vec<EngineEvent>, SessionError> {
        let Self { map, engine, .. } = self;
        let state = map.get_mut(id)?;
        let mut events = Vec::new();
        while let Some(chunk) = state.pop_inbound() {
            events.extend(engine.push_audio(state, &chunk)?);
        }
        Ok(events)
    }

    /// Enqueue PCM16 audio and process the mailbox; return emitted events.
    pub fn push_audio(
        &mut self,
        id: SessionId,
        pcm16: &[i16],
    ) -> Result<Vec<EngineEvent>, SessionError> {
        self.enqueue_audio(id, pcm16)?;
        self.process_inbound(id)
    }

    /// Finalize a session: flush the engine, remove it from the map, return events.
    pub fn finalize(&mut self, id: SessionId) -> Result<Vec<EngineEvent>, SessionError> {
        let Self { map, engine, .. } = self;
        let mut state = map.remove(id)?;
        Ok(engine.finalize(&mut state)?)
    }

    /// Cancel a session: drop state with no transcript events.
    pub fn cancel(&mut self, id: SessionId) -> Result<(), SessionError> {
        self.map.remove(id)?;
        Ok(())
    }

    /// Close a session: free the map entry and drop stream state.
    pub fn close(&mut self, id: SessionId) -> Result<(), SessionError> {
        self.cancel(id)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{SessionError, SessionId, SessionMap, Sessions};
    use crate::engine::{EngineEvent, MockEngine};

    #[test]
    fn create_then_remove() {
        let mut map = SessionMap::new();
        let id = map.create(16);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(id).unwrap().session_id(), id);

        let state = map.remove(id).unwrap();
        assert_eq!(state.session_id(), id);
        assert!(map.is_empty());
        assert_eq!(map.get(id).unwrap_err(), SessionError::Unknown(id));
    }

    #[test]
    fn unknown_id_errors() {
        let mut map = SessionMap::new();
        let id = SessionId::from_raw(42);
        assert_eq!(map.get(id).unwrap_err(), SessionError::Unknown(id));
        assert_eq!(map.remove(id).unwrap_err(), SessionError::Unknown(id));
    }

    #[test]
    fn create_allocates_distinct_ids() {
        let mut map = SessionMap::new();
        let a = map.create(16);
        let b = map.create(16);
        assert_ne!(a, b);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn open_allocates_stream_and_registers() {
        let mut sessions = Sessions::new(MockEngine::new());
        assert!(sessions.is_empty());

        let id = sessions.open();
        assert_eq!(sessions.len(), 1);

        let state = sessions.get(id).unwrap();
        assert_eq!(state.session_id(), id);
        assert_eq!(state.chunks_pushed(), 0);
    }

    #[test]
    fn push_audio_returns_mock_partial() {
        let mut sessions = Sessions::new(MockEngine::new());
        let id = sessions.open();

        let events = sessions.push_audio(id, &[0, 1, 2]).unwrap();
        assert_eq!(
            events,
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        assert_eq!(sessions.get(id).unwrap().chunks_pushed(), 1);
    }

    #[test]
    fn push_audio_unknown_session() {
        let mut sessions = Sessions::new(MockEngine::new());
        let id = SessionId::from_raw(9);
        assert_eq!(
            sessions.push_audio(id, &[]).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn finalize_returns_final_and_removes_session() {
        let mut sessions = Sessions::new(MockEngine::new());
        let id = sessions.open();
        sessions.push_audio(id, &[0]).unwrap();

        let events = sessions.finalize(id).unwrap();
        assert_eq!(
            events,
            vec![EngineEvent::Final {
                text: "final-1".into()
            }]
        );
        assert!(sessions.is_empty());
        assert_eq!(sessions.get(id).unwrap_err(), SessionError::Unknown(id));
    }

    #[test]
    fn cancel_drops_session_without_events() {
        let mut sessions = Sessions::new(MockEngine::new());
        let id = sessions.open();
        sessions.push_audio(id, &[0]).unwrap();

        sessions.cancel(id).unwrap();
        assert!(sessions.is_empty());
        assert_eq!(sessions.get(id).unwrap_err(), SessionError::Unknown(id));
        assert_eq!(
            sessions.push_audio(id, &[]).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn duplicate_finalize_after_remove_errors() {
        let mut sessions = Sessions::new(MockEngine::new());
        let id = sessions.open();
        sessions.finalize(id).unwrap();
        assert_eq!(
            sessions.finalize(id).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn duplicate_cancel_after_remove_errors() {
        let mut sessions = Sessions::new(MockEngine::new());
        let id = sessions.open();
        sessions.cancel(id).unwrap();
        assert_eq!(
            sessions.cancel(id).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn cancel_after_finalize_errors() {
        let mut sessions = Sessions::new(MockEngine::new());
        let id = sessions.open();
        sessions.finalize(id).unwrap();
        assert_eq!(
            sessions.cancel(id).unwrap_err(),
            SessionError::Unknown(id)
        );
    }

    #[test]
    fn close_frees_map_entry_and_drops_state() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut sessions = Sessions::new(MockEngine::new());
        let id = sessions.open();
        sessions
            .get_mut(id)
            .unwrap()
            .track_drops(Arc::clone(&drops));

        sessions.close(id).unwrap();
        assert!(sessions.is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_sessions_drops_open_stream_state() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut sessions = Sessions::new(MockEngine::new());
            let a = sessions.open();
            let b = sessions.open();
            sessions
                .get_mut(a)
                .unwrap()
                .track_drops(Arc::clone(&drops));
            sessions
                .get_mut(b)
                .unwrap()
                .track_drops(Arc::clone(&drops));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn enqueue_busy_when_mailbox_full() {
        let mut sessions = Sessions::with_mailbox_capacity(MockEngine::new(), 1);
        let id = sessions.open();

        sessions.enqueue_audio(id, &[1]).unwrap();
        assert_eq!(sessions.get(id).unwrap().mailbox_len(), 1);
        assert_eq!(
            sessions.enqueue_audio(id, &[2]).unwrap_err(),
            SessionError::Busy
        );

        let events = sessions.process_inbound(id).unwrap();
        assert_eq!(
            events,
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        sessions.enqueue_audio(id, &[3]).unwrap();
    }

    #[test]
    fn two_sessions_interleaved_no_crosstalk() {
        let mut sessions = Sessions::new(MockEngine::new());
        let a = sessions.open();
        let b = sessions.open();
        assert_ne!(a, b);

        assert_eq!(
            sessions.push_audio(a, &[0]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        assert_eq!(
            sessions.push_audio(b, &[0]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-1".into()
            }]
        );
        assert_eq!(
            sessions.push_audio(a, &[0]).unwrap(),
            vec![EngineEvent::Partial {
                text: "partial-2".into()
            }]
        );

        assert_eq!(sessions.get(a).unwrap().chunks_pushed(), 2);
        assert_eq!(sessions.get(b).unwrap().chunks_pushed(), 1);

        assert_eq!(
            sessions.finalize(b).unwrap(),
            vec![EngineEvent::Final {
                text: "final-1".into()
            }]
        );
        assert_eq!(
            sessions.finalize(a).unwrap(),
            vec![EngineEvent::Final {
                text: "final-2".into()
            }]
        );
        assert!(sessions.is_empty());
    }
}
