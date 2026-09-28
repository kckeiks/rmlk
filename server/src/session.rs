//! Per-call streaming session state.

use std::collections::HashMap;

use thiserror::Error;

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
}

impl StreamState {
    /// Create state for `session_id` with no audio pushed yet.
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            chunks_pushed: 0,
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
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new(SessionId::default())
    }
}

/// Session lookup failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SessionError {
    #[error("unknown session {0:?}")]
    Unknown(SessionId),
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

    /// Allocate a new session id, insert default state, and return the id.
    pub fn create(&mut self) -> SessionId {
        let id = SessionId::from_raw(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.sessions.insert(id, StreamState::new(id));
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
}

impl<E> Sessions<E> {
    /// Create a session registry backed by `engine`.
    pub fn new(engine: E) -> Self {
        Self {
            map: SessionMap::new(),
            engine,
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

impl<E: crate::engine::Engine> Sessions<E> {
    /// Open a session: allocate `StreamState`, register it, return its id.
    pub fn open(&mut self) -> SessionId {
        self.map.create()
    }

    /// Borrow session state by id.
    pub fn get(&self, id: SessionId) -> Result<&StreamState, SessionError> {
        self.map.get(id)
    }

    /// Borrow session state by id, mutably.
    pub fn get_mut(&mut self, id: SessionId) -> Result<&mut StreamState, SessionError> {
        self.map.get_mut(id)
    }
}

#[cfg(test)]
mod tests {
    use super::{SessionError, SessionId, SessionMap, Sessions};
    use crate::engine::MockEngine;

    #[test]
    fn create_then_remove() {
        let mut map = SessionMap::new();
        let id = map.create();
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
        let a = map.create();
        let b = map.create();
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
}
