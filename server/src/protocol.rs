//! Wire protocol between clients and this server.
//!
//! Frames (planned): open session, audio chunk, partial, final, cancel,
//! finalize, error. Concrete codec (WebSocket binary / length-prefixed) lands
//! in Phase 1 with tests first.
