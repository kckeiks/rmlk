use crate::attributes::Attribute;
use crate::model::StringStringEntryProto;

pub struct Node {
    input: Vec<String>,
    output: Vec<String>,
    name: Option<String>,
    op_type: Option<String>,
    domain: Option<String>,
    overload: Option<String>,
    attribute: Vec<Attribute>,
    doc_string: Option<String>,
    metadata_props: Vec<StringStringEntryProto>,
}
