//! Per-call streaming session state.

/// Opaque handle for a live streaming call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SessionId(pub u64);

/// Per-call streaming state.
#[derive(Debug, Default)]
pub struct StreamState {
    pub session_id: SessionId,
}
