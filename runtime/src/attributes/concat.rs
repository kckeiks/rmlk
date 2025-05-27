use rmlk_schema::Attribute;
use std::collections::HashMap;

pub fn get_axis(attrs: &HashMap<Box<str>, Attribute>) -> Option<i32> {
    attrs.get("axis")?.int()
}
