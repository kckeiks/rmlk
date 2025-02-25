use rmlk_schema::{Attribute, DataType};
use std::collections::HashMap;

pub fn _get_value_i32(attrs: &HashMap<Box<str>, Attribute>) -> Option<i32> {
    attrs.get("value")?.int()
}

pub fn get_value_f32(attrs: &HashMap<Box<str>, Attribute>) -> Option<f32> {
    attrs.get("value")?.float()
}

pub fn get_dtype(attrs: &HashMap<Box<str>, Attribute>) -> Option<DataType> {
    attrs.get("dtype")?.dtype()
}
