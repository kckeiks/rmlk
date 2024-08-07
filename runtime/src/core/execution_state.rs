use crate::core::instance_state::ModelInstanceState;
use crate::core::tensor::Tensor;
use crate::core::values::Values;
use crate::core::DeviceService;
use crate::{Error, Result};
use rmlk_graph::Node;
use rmlk_ir::Op;
use std::collections::HashMap;
use std::sync::Arc;

/// The execution context.
///
/// This object provides access to the values
/// needed for all computations in the graph.
pub struct ExecutionState<T: DeviceService> {
    /// All the tensors for the entire computation graph.
    ///
    /// This includes the inputs, outputs and
    /// intermediate values of the entire graph.
    values: Values<T::Data>,
    /// Indices for all of a node's tensors.
    ///
    /// The order is inputs, optional inputs and outputs.
    node_values: Box<[usize]>,
    node_to_index_map: HashMap<usize, usize>,
    /// Reference to the session state.
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
        let mut index_to_tensor_index = HashMap::new();
        let mut node_tensors = Vec::with_capacity(3 * graph.nodes().count());

        // Todo: Remove when we have a plan with steps to traverse the graph.
        for (node_id, node) in graph.nodes().enumerate() {
            // We already loaded the initializers.
            if graph.get_initial_tensor(node_id).is_some()
                || matches!(node.op(), Op::Const | Op::NoOp)
            {
                continue;
            }

            let index = node_tensors.len();
            index_to_tensor_index.insert(node_id, index);

            for input in node.inputs() {
                debug_assert!(values.get(*input).is_some());

                if graph.get_node(*input).is_some() {
                    node_tensors.push(*input);
                } else {
                    return Err(Error::MissingNode);
                }
            }

            for output in node.outputs() {
                debug_assert!(values.get(*output).is_some());

                if graph.get_node(*output).is_some() {
                    node_tensors.push(*output);
                } else {
                    return Err(Error::MissingNode);
                }
            }
        }

        Ok(Self {
            values,
            node_to_index_map: index_to_tensor_index,
            node_values: node_tensors.into_boxed_slice(),
            instance_state,
        })
    }

    #[cfg(test)]
    pub fn test_new(
        values: Values<T::Data>,
        node_values: Box<[usize]>,
        node_to_index_map: HashMap<usize, usize>,
        instance_state: Arc<ModelInstanceState<T>>,
    ) -> Self {
        Self {
            values,
            node_values,
            node_to_index_map,
            instance_state,
        }
    }

    pub fn get_node(&self, node_id: usize) -> Option<&Node> {
        self.instance_state.graph().get_node(node_id)
    }

    pub fn get_input_count(&self, node_id: usize) -> Option<usize> {
        let node = self.get_node(node_id)?;
        Some(node.inputs().len())
    }

    pub fn get_output_count(&self, node_id: usize) -> Option<usize> {
        let node = self.get_node(node_id)?;
        Some(node.outputs().len())
    }

    pub fn get_tensor(&self, node_index: usize) -> Option<&Tensor<T::Data>> {
        let tensor_index = self.get_tensor_index(node_index)?;
        self.values.get(tensor_index)
    }

    pub fn get_tensor_mut(&mut self, node_index: usize) -> Option<&mut Tensor<T::Data>> {
        let tensor_index = self.get_tensor_index(node_index)?;
        self.values.get_mut(tensor_index)
    }

    pub fn get_value(&mut self, node_index: usize) -> Option<&mut Tensor<T::Data>> {
        self.values.get_mut(node_index)
    }

    fn get_tensor_index(&self, node_index: usize) -> Option<usize> {
        self.node_values.get(node_index).copied()
    }

    pub fn get_value_index(&self, node_id: &usize) -> Option<usize> {
        self.node_to_index_map.get(node_id).copied()
    }
}
