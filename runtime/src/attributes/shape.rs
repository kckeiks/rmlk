use rmlk_schema::Attribute;
use std::collections::HashMap;

pub fn get_start(attrs: &HashMap<Box<str>, Attribute>) -> i32 {
    attrs.get("start").and_then(|attr| attr.int()).unwrap_or(0)
}

pub fn get_end(attrs: &HashMap<Box<str>, Attribute>) -> Option<i32> {
    attrs.get("end").and_then(|attr| attr.int())
}
