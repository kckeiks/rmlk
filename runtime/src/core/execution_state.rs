use crate::core::instance_state::ModelInstanceState;
use crate::core::tensor::Tensor;
use crate::core::values::Values;
use crate::core::DeviceService;
use crate::{Error, Result};
use log::trace;
use rmlk_graph::Node;
use rmlk_schema::Op;
use std::collections::HashMap;
use std::sync::Arc;

/// Execution state for a computational graph.
///
/// This object provides access to the values
/// needed for all computations in the graph.
pub struct ExecutionState<T: DeviceService> {
    /// All the tensor values for the entire computation graph.
    ///
    /// This includes the inputs, outputs and
    /// intermediate values of the entire graph.
    values: Values<T::Data>,
    /// Value indices for finding an operation's tensor values.
    ///
    /// The order is inputs, optional inputs and outputs.
    op_values: Box<[usize]>,
    /// Maps a computation given by a node ID
    /// to the start of the node's values in `values`.
    node_to_value_index_map: HashMap<usize, usize>,
    /// Reference to the model instance state.
    instance_state: Arc<ModelInstanceState<T>>,
}

impl<T> ExecutionState<T>
where
    T: DeviceService,
{
    pub fn new(
        instance_state: Arc<ModelInstanceState<T>>,
        values: Values<T::Data>,
    ) -> Result<ExecutionState<T>> {
        let graph = instance_state.graph();
        let mut node_to_value_index_map = HashMap::new();
        let mut node_values = Vec::with_capacity(3 * graph.nodes().count());

        // Todo: Remove when we have a plan with steps to traverse the graph.
        for (node_id, node) in graph.nodes().enumerate() {
            // We already loaded the initializers.
            if graph.get_initial_tensor(node_id).is_some()
                || matches!(node.def().op(), Op::Const | Op::NoOp)
            {
                continue;
            }

            let index = node_values.len();
            node_to_value_index_map.insert(node_id, index);

            for input in node.inputs() {
                debug_assert!(values.get(*input).is_some());

                if graph.get_node(*input).is_some() {
                    node_values.push(*input);
                } else {
                    return Err(Error::MissingNode);
                }
            }

            for output in node.outputs() {
                debug_assert!(values.get(*output).is_some());

                if graph.get_node(*output).is_some() {
                    node_values.push(*output);
                } else {
                    return Err(Error::MissingNode);
                }
            }
        }

        trace!("node_values={:?}", node_values);
        trace!("node_to_value_index_map={:?}", node_to_value_index_map);

        Ok(Self {
            values,
            node_to_value_index_map,
            op_values: node_values.into_boxed_slice(),
            instance_state,
        })
    }

    /// Get a reference to the node.
    pub fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.instance_state.graph().get_node(node_id)
    }

    /// Get a count of all the inputs for the given node.
    pub fn get_input_count(&self, node_id: usize) -> Option<usize> {
        let node = self.get_node(node_id)?;
        Some(node.inputs().len())
    }

    /// Get a count of all the outputs for the given node.
    pub fn get_output_count(&self, node_id: usize) -> Option<usize> {
        let node = self.get_node(node_id)?;
        Some(node.outputs().len())
    }

    /// Get the shared tensor value.
    ///
    /// The value index for a given computation can be
    /// found using [`ExecutionState::get_value_index`].
    pub fn get_value(&self, value_index: usize) -> Option<&Tensor<T::Data>> {
        let index = self.get_inner_index(value_index)?;
        self.values.get(index)
    }

    /// Get the mutable tensor value.
    ///
    /// The value index for a given computation can be
    /// found using [`ExecutionState::get_value_index`].
    pub fn get_value_mut(&mut self, value_index: usize) -> Option<&mut Tensor<T::Data>> {
        let index = self.get_inner_index(value_index)?;
        self.values.get_mut(index)
    }

    /// Get the tensor value given a node ID.
    pub fn get_value_from_node_id_mut(&mut self, node_id: usize) -> Option<&mut Tensor<T::Data>> {
        self.values.get_mut(node_id)
    }

    /// Get the starting index for the values of a node.
    pub fn get_value_index(&self, node_id: &usize) -> Option<usize> {
        self.node_to_value_index_map.get(node_id).copied()
    }

    /// Get the index of the actual value.
    fn get_inner_index(&self, value_index: usize) -> Option<usize> {
        self.op_values.get(value_index).copied()
    }
}
