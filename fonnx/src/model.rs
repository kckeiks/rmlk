use crate::graph::Graph;
use crate::op::{Function, OperatorSetId};

pub enum Version {
    Ir2024 = 1,
}

pub struct Model {
    ir_version: Version,
    opset_import: Vec<OperatorSetId>,
    producer_name: String,
    producer_version: String,
    domain: Option<String>,
    model_version: Option<i64>,
    doc_string: Option<String>,
    graph: Option<Graph>,
    metadata_props: Vec<StringStringEntryProto>,
    functions: Vec<Function>,
}

pub struct StringStringEntryProto {
    key: Option<String>,
    value: Option<String>,
}

pub struct TensorAnnotation {
    tensor_name: Option<String>,
    quant_parameter_tensor_names: Vec<StringStringEntryProto>,
}
