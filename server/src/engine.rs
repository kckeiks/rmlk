//! Inference backend trait.
//!
//! Protocol and sessions talk only to this trait so backend types stay out of
//! the wire API.

/// Placeholder until Phase 1 introduces a mock engine and Phase 2 a real one.
pub trait Engine: Send {
    // Methods arrive with the phases that need them.
}
