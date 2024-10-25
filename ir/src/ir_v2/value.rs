use crate::{DataType, StringStringEntryProto, Tensor};
use serde::{Deserialize, Serialize};

/// Defines information on value, including the name, the type, and
/// the shape of the value.
#[derive(Debug, Deserialize, Serialize)]
pub struct ValueInfo {
    /// This field MUST be present in this version of the IR.
    pub id: usize,
    /// This field MUST be present in this version of the IR for
    /// inputs and outputs of the top-level graph.
    pub ty: Option<TypeValue>,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum TypeValue {
    Tensor { ty: i32, dims: Vec<usize> },
}

impl TypeValue {
    pub fn ty(&self) -> i32 {
        match self {
            TypeValue::Tensor { ty, .. } => *ty,
        }
    }

    pub fn dims(&self) -> &Vec<usize> {
        match self {
            TypeValue::Tensor { dims, .. } => &dims,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Value {
    pub dtype: DataType,
    pub dims: Vec<usize>,
    pub name: String,
}

impl Value {
    pub fn new(dtype: DataType, dims: Vec<usize>, name: String) -> Self {
        Self { name, dtype, dims }
    }
}
