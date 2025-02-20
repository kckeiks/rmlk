use std::collections::HashMap;
use rmlk_schema::Attribute;

pub fn get_value(attrs: &HashMap<Box<str>, Attribute>) -> Option<i32> {
    attrs.get("value")?.int()
}
