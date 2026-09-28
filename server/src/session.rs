//! Per-call streaming session and RNNT / transcript state ownership.
//!
//! Each live call owns a [`StreamState`]. The model engine is shared; state
//! is not. Details land with Phase 1 (lifecycle) and Phase 2 (real state).

/// Opaque handle for a live streaming call. Filled in during Phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

/// Per-call streaming state. Empty until the engine phase defines fields.
#[derive(Debug, Default)]
pub struct StreamState {
    pub session_id: SessionId,
}

impl Default for SessionId {
    fn default() -> Self {
        Self(0)
    }
}
