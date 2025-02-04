use rmlk_schema::Attribute;
use std::collections::HashMap;

pub fn get_axis(attrs: &HashMap<Box<str>, Attribute>) -> i32 {
    attrs.get("axis").and_then(|attr| attr.int()).unwrap_or(0)
}
