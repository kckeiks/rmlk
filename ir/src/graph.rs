use crate::attributes::ValueInfo;
use crate::error::Error;
use crate::model::{StringStringEntryProto, TensorAnnotation};
use crate::node::Node;
use crate::onnx::GraphProto;
use crate::tensor::{SparseTensor, Tensor};
use serde::{Deserialize, Serialize};

/// Graphs
///
/// A graph defines the computational logic of a model and is comprised of a parameterized
/// list of nodes that form a directed acyclic graph based on their inputs and outputs.
/// This is the equivalent of the "network" or "graph" in many deep learning
/// frameworks.
#[derive(Debug, Deserialize, Serialize)]
pub struct Graph {
    /// The nodes in the graph, sorted topologically.
    pub node: Vec<Node>,
    /// The name of the graph.
    pub name: Option<String>,
    /// A list of named tensor values, used to specify constant inputs of the graph.
    /// Each initializer (both TensorProto as well SparseTensorProto) MUST have a name.
    /// The name MUST be unique across both initializer and sparse_initializer,
    /// but the name MAY also appear in the input list.
    pub initializer: Vec<Tensor>,
    /// Initializers stored in sparse format.
    pub sparse_initializer: Vec<SparseTensor>,
    /// A human-readable documentation for this graph. Markdown is allowed.
    pub doc_string: Option<String>,
    /// The inputs of the graph.
    pub input: Vec<ValueInfo>,
    /// The outputs of the graph.
    pub output: Vec<ValueInfo>,
    /// Information for the values in the graph. The ValueInfoProto.name's
    /// must be distinct. It is optional for a value to appear in value_info list.
    pub value_info: Vec<ValueInfo>,
    /// This field carries information to indicate the mapping among a tensor and its
    /// quantization parameter tensors. For example:
    /// For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated,
    /// which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.
    pub quantization_annotation: Vec<TensorAnnotation>,
    /// Named metadata values; keys should be distinct.
    pub metadata_props: Vec<StringStringEntryProto>,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            node: vec![],
            name: None,
            initializer: vec![],
            sparse_initializer: vec![],
            doc_string: None,
            input: vec![],
            output: vec![],
            value_info: vec![],
            quantization_annotation: vec![],
            metadata_props: vec![],
        }
    }
}

impl TryFrom<GraphProto<'_>> for Graph {
    type Error = Error;

    fn try_from(value: GraphProto) -> Result<Self, Self::Error> {
        let mut node = Vec::new();
        for n in value.node {
            node.push(n.try_into()?);
        }

        let mut initializer = Vec::new();
        for tensor in value.initializer {
            initializer.push(Tensor::from_onnx_tensor(tensor)?);
        }

        let mut sparse_initializer = Vec::new();
        for tensor in value.sparse_initializer {
            sparse_initializer.push(tensor.try_into()?);
        }

        let mut input = Vec::new();
        for info in value.input {
            input.push(info.try_into()?);
        }

        let mut output = Vec::new();
        for info in value.output {
            output.push(info.try_into()?);
        }

        let mut value_info = Vec::new();
        for info in value.value_info {
            value_info.push(info.try_into()?);
        }

        let mut quantization_annotation = Vec::new();
        for annotation in value.quantization_annotation {
            quantization_annotation.push(annotation.into());
        }

        let mut metadata_props = Vec::new();
        for prop in value.metadata_props {
            metadata_props.push(prop.into());
        }

        Ok(Self {
            node,
            name: value.name.map(|str| str.to_string()),
            initializer,
            sparse_initializer,
            doc_string: value.doc_string.map(|str| str.to_string()),
            input,
            output,
            value_info,
            quantization_annotation,
            metadata_props,
        })
    }
}
