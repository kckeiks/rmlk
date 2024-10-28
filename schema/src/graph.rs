use crate::node::Node;
use crate::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub initializer: HashMap<usize, Tensor>,
    /// The inputs of the graph.
    pub input: Vec<usize>,
    /// The outputs of the graph.
    pub output: Vec<usize>,
    /// This field carries information to indicate the mapping among a tensor and its
    /// quantization parameter tensors. For example:
    /// For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated,
    /// which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.
    pub quantization_annotation: Option<HashMap<u64, HashMap<String, String>>>,
}
