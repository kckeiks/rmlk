//! GPU work queue: ready streams, microbatch gather/scatter, admission.
//!
//! Phase 4: time-multiplex concurrent sessions (B=1 runs).
//! Phase 6: true cross-call batching if a B>1 ONNX export exists.
