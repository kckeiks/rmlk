use serde::{Deserialize, Serialize};
use crate::ir_v2::tensor::Tensor;

/// Attributes
///
/// A named attribute containing either singular float, integer, string, graph,
/// and tensor values, or repeated float, integer, string, graph, and tensor values.
/// An AttributeProto MUST contain the name field, and *only one* of the
/// following content fields, effectively enforcing a C/C++ union equivalent.
#[derive(Debug, Deserialize, Serialize)]
pub struct Attribute {
    /// The name of the attribute.
    pub name: u32,
    /// If ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
    /// In this case, this AttributeProto does not contain data, and it's a reference of attribute
    /// in parent scope.
    /// NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
    pub ref_attr_name: Option<u32>,
    /// The type of the attribute.
    pub ty: AttributeType,
    /// A human-readable documentation for this attribute. Markdown is allowed.
    pub doc_string: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum AttributeType {
    Float(f32),
    Int(i32),
    String(Vec<u8>),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Ints(Vec<i32>),
    Strings(Vec<Vec<u8>>),
    Tensors(Vec<Tensor>),
}
