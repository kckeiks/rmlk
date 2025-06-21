use rmlk_schema::Attribute;
use std::collections::HashMap;

pub fn get_allow_zero(attrs: &HashMap<Box<str>, Attribute>) -> bool {
    match attrs.get("allowzero").and_then(|attr| attr.int()) {
        Some(0) => false,
        Some(_) => true,
        _ => false,
    }
}
