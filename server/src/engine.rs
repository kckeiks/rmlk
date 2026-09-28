//! Inference backend trait.
//!
//! Protocol and sessions talk only to this trait so backend types stay out of
//! the wire API.

pub trait Engine: Send {}
