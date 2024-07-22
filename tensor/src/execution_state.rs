use crate::tensor::Tensor;
use rmlk_graph::{Graph, Node};
use std::sync::Arc;

/// The execution context.
///
/// This object provides access to the values
/// needed for all computations in the graph.
pub struct ExecutionState<T> {
    /// All the tensors for the entire computation graph.
    ///
    /// This includes the inputs, outputs and
    /// intermediate values of the entire graph.
    tensors: Box<[Tensor<T>]>,
    /// Indices for all of a node's tensors.
    ///
    /// The order is inputs, optional inputs and outputs.
    node_tensors: Box<[usize]>,
    graph: Arc<Graph>,
}

impl<T> ExecutionState<T> {
    pub fn new(
        graph: Arc<Graph>,
        tensors: Box<[Tensor<T>]>,
        node_tensors: Box<[usize]>,
    ) -> ExecutionState<T> {
        Self {
            tensors,
            node_tensors,
            graph,
        }
    }

    pub fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.graph.get_node(node_id)
    }

    pub fn get_input_count(&self, node_id: usize) -> Option<usize> {
        let node = self.get_node(node_id)?;
        Some(node.inputs().len())
    }

    pub fn get_output_count(&self, node_id: usize) -> Option<usize> {
        let node = self.get_node(node_id)?;
        Some(node.outputs().len())
    }

    pub fn get_tensor(&self, node_index: usize) -> Option<&Tensor<T>> {
        let tensor_index = self.get_tensor_index(node_index)?;
        self.tensors.get(tensor_index)
    }

    pub fn get_tensor_mut(&mut self, node_index: usize) -> Option<&mut Tensor<T>> {
        let tensor_index = self.get_tensor_index(node_index)?;
        self.tensors.get_mut(tensor_index)
    }

    fn get_tensor_index(&self, node_index: usize) -> Option<usize> {
        self.node_tensors.get(node_index).copied()
    }
}
