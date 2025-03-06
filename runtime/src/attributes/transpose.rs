use rmlk_schema::Attribute;
use std::collections::HashMap;

pub fn get_perm(attrs: &HashMap<Box<str>, Attribute>) -> Option<&[i32]> {
    attrs.get("perm")?.ints()
}
