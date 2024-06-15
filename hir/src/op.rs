use crate::attributes;
use crate::attributes::ValueInfo;
use crate::model::StringStringEntryProto;
use crate::node::Node;

pub struct OperatorSetId {
    domain: Option<String>,
    version: i64,
}

pub struct Function {
    name: Option<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attribute: Vec<Attribute>,
    node: Vec<Node>,
    doc_string: Option<String>,
    opset_import: Vec<OperatorSetId>,
    domain: Option<String>,
    overload: Option<String>,
    value_info: Vec<ValueInfo>,
    metadata_props: Vec<StringStringEntryProto>,
}

enum Attribute {
    String { value: Vec<String> },
    Object { value: attributes::Attribute },
}
