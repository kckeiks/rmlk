use crate::tensor::Tensor;
use rmlk_graph::{Graph, Node};
use std::sync::Arc;

/// The execution context.
///
/// This object provides access to the values
/// needed for all computations in the graph.
pub struct ExecutionState<T> {
    /// All the values for the entire graph.
    ///
    /// This includes the inputs, outputs and
    /// intermediate values of the entire graph.
    /// For each entry, the order is `inputs` then `outputs`.
    tensors: Vec<Tensor<T>>,
    /// Offset from where the node's inputs & outputs region begins in `tensors`.
    offsets: Vec<usize>,
    /// Minimum index value for all the nodes in the graph for this context.
    _min_value: usize,
    _graph: Arc<Graph>,
}

impl<T> ExecutionState<T> {
    pub fn new(
        graph: Arc<Graph>,
        tensors: Vec<Tensor<T>>,
        offsets: Vec<usize>,
    ) -> ExecutionState<T> {
        Self {
            tensors,
            offsets,
            _graph: graph,
            // Todo: compute from graph.
            _min_value: 0,
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self._graph.get_node(id)
    }

    pub fn get_input_count(&self, node_id: usize) -> Option<usize> {
        let node = self.get_node(node_id)?;
        Some(node.inputs().len())
    }

    pub fn get_tensor(&self, node_id: usize, offset: usize) -> Option<&Tensor<T>> {
        let index = self.offsets.get(node_id)?;
        self.tensors.get(index.checked_add(offset)?)
    }

    pub fn get_tensor_mut(&mut self, node_id: usize, offset: usize) -> Option<&mut Tensor<T>> {
        let index = self.offsets.get(node_id)?;
        self.tensors.get_mut(index.checked_add(offset)?)
    }
}
