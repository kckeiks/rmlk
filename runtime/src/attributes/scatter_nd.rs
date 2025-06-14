use crate::core::error::{InternalError, Result};
use rmlk_schema::Attribute;
use std::collections::HashMap;

pub enum Reduction {
    Add,
    Mul,
    Max,
    Min,
}

pub fn get_reduction(attrs: &HashMap<Box<str>, Attribute>) -> Result<Option<Reduction>> {
    let bytes = match attrs.get("reduction").and_then(|attr| attr.string()) {
        None => return Ok(None),
        Some(b) => b,
    };

    match String::from_utf8_lossy(bytes).as_ref() {
        "none" | "None" => Ok(None),
        "add" | "Add" => Ok(Some(Reduction::Add)),
        "mul" | "Mul" => Ok(Some(Reduction::Mul)),
        "max" | "Max" => Ok(Some(Reduction::Max)),
        "min" | "Min" => Ok(Some(Reduction::Min)),
        val => Err(InternalError::InvalidAttribute {
            name: format!("unknown value `{val}` for `reduction`"),
        }),
    }
}
