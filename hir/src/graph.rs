use crate::attributes::ValueInfo;
use crate::model::{StringStringEntryProto, TensorAnnotation};
use crate::node::Node;
use crate::tensor::{SparseTensor, Tensor};

pub struct Graph {
    pub node: Vec<Node>,
    pub name: Option<String>,
    // A list of named tensor values, used to specify constant inputs of the graph.
    // Each initializer (both TensorProto as well SparseTensorProto) MUST have a name.
    // The name MUST be unique across both initializer and sparse_initializer,
    // but the name MAY also appear in the input list.
    pub initializer: Vec<Tensor>,
    pub sparse_initializer: Vec<SparseTensor>,
    pub doc_string: Option<String>,
    pub input: Vec<ValueInfo>,
    pub output: Vec<ValueInfo>,
    // Information for the values in the graph. The ValueInfoProto.name's
    // must be distinct. It is optional for a value to appear in value_info list.
    pub value_info: Vec<ValueInfo>,
    pub quantization_annotation: Vec<TensorAnnotation>,
    pub metadata_props: Vec<StringStringEntryProto>,
}
