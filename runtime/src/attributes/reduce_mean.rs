use rmlk_schema::Attribute;
use std::collections::HashMap;

pub fn get_keep_dims(attrs: &HashMap<Box<str>, Attribute>) -> bool {
    let value = attrs
        .get("keepdims")
        .and_then(|attr| attr.int())
        .unwrap_or(1);
    value > 0
}

pub fn get_noop_with_empty_axes(attrs: &HashMap<Box<str>, Attribute>) -> bool {
    let value = attrs
        .get("noop_with_empty_axes")
        .and_then(|attr| attr.int())
        .unwrap_or(0);
    value > 0
}

pub fn get_axes(attrs: &HashMap<Box<str>, Attribute>) -> Option<&[i32]> {
    attrs.get("axes").and_then(|attr| attr.ints())
}
