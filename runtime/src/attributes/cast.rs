use crate::attributes::error::AttributeError;
use crate::attributes::error::Result;
use rmlk_schema::{Attribute, DataType, Op};
use std::collections::HashMap;

pub fn get_value(attrs: &HashMap<Box<str>, Attribute>) -> Result<Option<DataType>> {
    match attrs.get("to") {
        None => Ok(None),
        Some(value) => {
            let dtype = value
                .int()
                .ok_or_else(|| AttributeError::InvalidDataType {
                    name: "to",
                    op: Op::Cast,
                    expected: DataType::Int32,
                    actual: value.dtype().unwrap_or(DataType::Undefined),
                })?
                .try_into()
                .map_err(|_| AttributeError::UnknownDataType {
                    name: "to",
                    op: Op::Cast,
                })?;
            Ok(Some(dtype))
        }
    }
}
