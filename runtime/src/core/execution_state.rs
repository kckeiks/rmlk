use crate::core::session_state::ModelInstanceState;
use crate::core::tensor::Tensor;
use crate::core::ExecutionProvider;
use rmlk_graph::Node;
use std::sync::Arc;

/// The execution context.
///
/// This object provides access to the values
/// needed for all computations in the graph.
pub struct ExecutionState<P: ExecutionProvider> {
    /// All the tensors for the entire computation graph.
    ///
    /// This includes the inputs, outputs and
    /// intermediate values of the entire graph.
    all_tensors: Box<[Option<Tensor<P::Data>>]>,
    /// Indices for all of a node's tensors.
    ///
    /// The order is inputs, optional inputs and outputs.
    node_tensors: Box<[usize]>,
    /// Reference to the session state.
    session_state: Arc<ModelInstanceState<P>>,
}

impl<P> ExecutionState<P>
where
    P: ExecutionProvider,
{
    pub fn new(
        session_state: Arc<ModelInstanceState<P>>,
        tensors: Box<[Option<Tensor<P::Data>>]>,
        node_tensors: Box<[usize]>,
    ) -> ExecutionState<P> {
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

    pub fn get_tensor(&self, node_index: usize) -> Option<&Tensor<P::Data>> {
        let tensor_index = self.get_tensor_index(node_index)?;
        self.all_tensors
            .get(tensor_index)
            .map(Option::as_ref)
            .flatten()
    }

    pub fn get_tensor_mut(&mut self, node_index: usize) -> Option<&mut Tensor<P::Data>> {
        let tensor_index = self.get_tensor_index(node_index)?;
        self.all_tensors
            .get_mut(tensor_index)
            .map(Option::as_mut)
            .flatten()
    }

    pub fn get_value(&mut self, node_index: usize) -> Option<&mut Tensor<P::Data>> {
        self.all_tensors
            .get_mut(node_index)
            .map(Option::as_mut)
            .flatten()
    }

    fn get_tensor_index(&self, node_index: usize) -> Option<usize> {
        self.node_tensors.get(node_index).copied()
    }

    pub fn _get_tensors(&self) -> &[Option<Tensor<P::Data>>] {
        self.all_tensors.as_ref()
    }

    pub fn _get_node_tensors(&self) -> &[usize] {
        self.node_tensors.as_ref()
    }
}
