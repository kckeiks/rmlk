use rmlk_schema::Attribute;
use std::collections::HashMap;

pub fn get_upper(attrs: &HashMap<Box<str>, Attribute>) -> bool {
    match attrs.get("upper").and_then(|attr| attr.int()) {
        Some(0) => false,
        _ => true,
    }
}
