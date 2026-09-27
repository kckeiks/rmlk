use rmlk_schema::{Attribute, DataType};
use std::collections::HashMap;

pub fn _contains_sparse_value(attrs: &HashMap<Box<str>, Attribute>) -> bool {
    attrs.get("sparse_value").is_some()
}

pub type RawTensorAttr<'a> = (DataType, &'a [usize], Option<&'a [u8]>);

pub fn get_raw_value(attrs: &HashMap<Box<str>, Attribute>) -> Option<RawTensorAttr<'_>> {
    attrs
        .get("value")?
        .tensor()
        .map(|t| (t.data_type, t.dims.as_slice(), t.raw_data.as_deref()))
}

pub fn get_float(attrs: &HashMap<Box<str>, Attribute>) -> Option<f32> {
    attrs.get("value_float")?.float()
}

pub fn get_floats(attrs: &HashMap<Box<str>, Attribute>) -> Option<&[f32]> {
    attrs.get("value_floats")?.floats()
}

pub fn get_int(attrs: &HashMap<Box<str>, Attribute>) -> Option<i32> {
    attrs.get("value_int")?.int()
}

pub fn get_ints(attrs: &HashMap<Box<str>, Attribute>) -> Option<&[i32]> {
    attrs.get("value_ints")?.ints()
}

pub fn _contains_string(attrs: &HashMap<Box<str>, Attribute>) -> bool {
    attrs.get("value_string").is_some()
}

pub fn _contains_strings(attrs: &HashMap<Box<str>, Attribute>) -> bool {
    attrs.get("value_strings").is_some()
}
