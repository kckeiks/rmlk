//! Per-call streaming session state.

use std::collections::HashMap;

use thiserror::Error;

/// Opaque handle for a live streaming call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SessionId(pub u64);

/// Per-call streaming state.
#[derive(Debug, Default)]
pub struct StreamState {
    pub session_id: SessionId,
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
        let id = SessionId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.sessions.insert(id, StreamState { session_id: id });
        id
    }

    /// Borrow session state by id.
    pub fn get(&self, id: SessionId) -> Result<&StreamState, SessionError> {
        self.sessions.get(&id).ok_or(SessionError::Unknown(id))
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

#[cfg(test)]
mod tests {
    use super::{SessionError, SessionId, SessionMap};

    #[test]
    fn create_then_remove() {
        let mut map = SessionMap::new();
        let id = map.create();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(id).unwrap().session_id, id);

        let state = map.remove(id).unwrap();
        assert_eq!(state.session_id, id);
        assert!(map.is_empty());
        assert_eq!(map.get(id).unwrap_err(), SessionError::Unknown(id));
    }

    #[test]
    fn unknown_id_errors() {
        let mut map = SessionMap::new();
        let id = SessionId(42);
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
}
