use crate::core::session_state::SessionState;
use crate::core::tensor::Tensor;
use rmlk_graph::Node;
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
    all_tensors: Box<[Tensor<T>]>,
    /// Indices for all of a node's tensors.
    ///
    /// The order is inputs, optional inputs and outputs.
    node_tensors: Box<[usize]>,
    /// Reference to the session state.
    session_state: Arc<SessionState>,
}

impl<T> ExecutionState<T> {
    pub fn new(
        session_state: Arc<SessionState>,
        tensors: Box<[Tensor<T>]>,
        node_tensors: Box<[usize]>,
    ) -> ExecutionState<T> {
        Self {
            all_tensors: tensors,
            node_tensors,
            session_state,
        }
    }

    pub fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.session_state.graph().get_node(node_id)
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
        self.all_tensors.get(tensor_index)
    }

    pub fn get_tensor_mut(&mut self, node_index: usize) -> Option<&mut Tensor<T>> {
        let tensor_index = self.get_tensor_index(node_index)?;
        self.all_tensors.get_mut(tensor_index)
    }

    pub fn get_value(&mut self, node_index: usize) -> Option<&mut Tensor<T>> {
        self.all_tensors.get_mut(node_index)
    }

    fn get_tensor_index(&self, node_index: usize) -> Option<usize> {
        self.node_tensors.get(node_index).copied()
    }

    pub fn get_tensors(&self) -> &[Tensor<T>] {
        self.all_tensors.as_ref()
    }

    pub fn get_node_tensors(&self) -> &[usize] {
        self.node_tensors.as_ref()
    }
}
