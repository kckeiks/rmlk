//! rmlk inference server.
//!
//! See `TODO.md` for the phased build plan.

pub mod engine;
pub mod protocol;
pub mod scheduler;
pub mod session;

/// Crate version from Cargo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
