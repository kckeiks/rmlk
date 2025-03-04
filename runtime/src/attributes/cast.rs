use crate::core::error::{InternalError, Result};
use rmlk_schema::{Attribute, DataType};
use std::collections::HashMap;

pub fn get_value(attrs: &HashMap<Box<str>, Attribute>) -> Result<Option<DataType>> {
    match attrs.get("to") {
        None => Ok(None),
        Some(value) => {
            let dtype = value
                .int()
                .ok_or_else(|| {
                    InternalError::InvalidAttributeDataType {
                        name: "expected the attribute to contain an integer that must be one of the types the data types in DataType".to_string()
                    }
                })?
                .try_into().map_err(|_| {
                    InternalError::InvalidAttributeDataType { name: "unknown data type value".to_string() }
                })?;
            Ok(Some(dtype))
        }
    }
}
