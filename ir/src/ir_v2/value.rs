use serde::{Deserialize, Serialize};
use crate::{StringStringEntryProto};

/// Defines information on value, including the name, the type, and
/// the shape of the value.
#[derive(Debug, Deserialize, Serialize)]
pub struct ValueInfo {
    /// This field MUST be present in this version of the IR.
    pub id: u64,
    /// This field MUST be present in this version of the IR for
    /// inputs and outputs of the top-level graph.
    pub ty: Option<TypeValue>,
    /// Optional name for debugging.
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum TypeValue {
    Tensor {
        ty: i32,
        dims: Vec<i64>,
    }
}