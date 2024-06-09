use crate::attributes::ValueInfo;
use crate::model::{StringStringEntryProto, TensorAnnotation};
use crate::node::Node;
use crate::tensor::{SparseTensor, Tensor};

pub struct Graph {
    node: Vec<Node>,
    name: Option<String>,
    initializer: Vec<Tensor>,
    sparse_initializer: Vec<SparseTensor>,
    doc_string: Option<String>,
    input: Vec<ValueInfo>,
    output: Vec<ValueInfo>,
    value_info: Vec<ValueInfo>,
    quantization_annotation: Vec<TensorAnnotation>,
    metadata_props: Vec<StringStringEntryProto>,
}
