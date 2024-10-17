use crate::DataType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data_type: DataType,
    pub raw_data: Option<Vec<u8>>,
    // Optional.
    pub name: Option<String>,
}
