use std::collections::HashMap;
use rmlk_schema::Attribute;

pub fn get_perm(attrs: &HashMap<Box<str>, Attribute>) -> Option<&[i32]> {
    attrs.get("perm")?.ints()
}