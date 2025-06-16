use crate::core::error::Result;
use crate::utils::FromBytes;
use half::f16;
use rmlk_schema::{Attribute, DataType};
use std::collections::HashMap;

pub enum AttributeTensor {
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl AttributeTensor {
    #[allow(dead_code)]
    pub fn data_type(&self) -> DataType {
        match self {
            AttributeTensor::F16(_) => DataType::Float16,
            AttributeTensor::F32(_) => DataType::Float,
            AttributeTensor::F64(_) => DataType::Double,
            AttributeTensor::I32(_) => DataType::Int32,
            AttributeTensor::I64(_) => DataType::Int64,
        }
    }
}

pub fn get_value(attrs: &HashMap<Box<str>, Attribute>) -> Result<Option<AttributeTensor>> {
    let tensor = match attrs.get("value").and_then(|a| a.tensor()) {
        None => return Ok(None),
        Some(tensor) => tensor,
    };

    let bytes = match tensor.raw_data.as_ref() {
        None => return Ok(None),
        Some(bytes) => bytes,
    };

    let data = match tensor.data_type {
        DataType::Float16 => AttributeTensor::F16(f16::from_bytes(bytes)?),
        DataType::Float => AttributeTensor::F32(f32::from_bytes(bytes)?),
        DataType::Double => AttributeTensor::F64(f64::from_bytes(bytes)?),
        DataType::Int32 => AttributeTensor::I32(i32::from_bytes(bytes)?),
        DataType::Int64 => AttributeTensor::I64(i64::from_bytes(bytes)?),
        _ => return Ok(None),
    };

    Ok(Some(data))
}
