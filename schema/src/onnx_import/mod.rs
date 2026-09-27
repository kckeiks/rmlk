//! ONNX model → [`crate::Graph`] conversion.
//!
//! `rocky` and the runtime builder call into this module; nothing outside
//! `rmlk-schema` should reimplement the name-to-builder walk.

mod transform;
mod traverse;

pub use transform::{graph_from_onnx_bytes, graph_from_onnx_proto, OnnxImportError};
pub use traverse::{visit_onnx, OnnxGraphTraverser, Result, TraversalError};
