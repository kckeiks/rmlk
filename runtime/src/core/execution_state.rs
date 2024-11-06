use crate::core::instance_state::ModelInstanceState;
use crate::core::store::TensorStore;
use crate::core::tensor::Tensor;
use crate::core::value::{InnerValue, Value};
use crate::core::DeviceService;
use crate::{Error, Result};
use log::trace;
use rmlk_graph::{Graph, Node};
use rmlk_schema::Op;
use rmlk_schema::{DataType, Definition};
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
    tensor: TensorStore<T::Data>,
    /// Tensor indices for finding an operation's tensor values.
    ///
    /// The order is inputs, optional inputs and outputs.
    op_tensors: Box<[usize]>,
    /// Maps a computation given by a node ID
    /// to the start of the node's values in `values`.
    node_to_tensor_index_map: HashMap<usize, usize>,
    /// Reference to the model instance state.
    instance_state: Arc<ModelInstanceState<T>>,
}

impl<T> ExecutionState<T>
where
    T: DeviceService,
{
    pub fn new(
        instance_state: Arc<ModelInstanceState<T>>,
        values: TensorStore<T::Data>,
    ) -> Result<ExecutionState<T>> {
        let graph = instance_state.graph();
        let mut node_to_value_index_map = HashMap::new();
        let mut node_values = Vec::with_capacity(3 * graph.nodes().count());

        // Todo: Remove when we have a plan with steps to traverse the graph.
        for (node_id, node) in graph.nodes().enumerate() {
            // We already loaded the initializers.
            if matches!(node.value().op(), Op::Const | Op::NoOp) {
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
            tensor: values,
            node_to_tensor_index_map: node_to_value_index_map,
            op_tensors: node_values.into_boxed_slice(),
            instance_state,
        })
    }

    /// Get a reference to the node.
    pub fn get_node(&self, node_id: usize) -> Option<&Node<Definition>> {
        self.instance_state.graph().get_node(node_id)
    }

    /// Get a shared tensor value.
    ///
    /// The value index for a given computation can be
    /// found using [`ExecutionState::get_tensor_index`].
    pub fn get_tensor(&self, value_index: usize) -> Option<&Tensor<T::Data>> {
        let index = self.get_inner_index(value_index)?;
        self.tensor.get(index)
    }

    /// Get a mutable tensor value.
    ///
    /// The value index for a given computation can be
    /// found using [`ExecutionState::get_tensor_index`].
    pub fn get_tensor_mut(&mut self, value_index: usize) -> Option<&mut Tensor<T::Data>> {
        let index = self.get_inner_index(value_index)?;
        self.tensor.get_mut(index)
    }

    /// Get the starting index for the values of a node.
    pub fn get_tensor_index(&self, node_id: &usize) -> Option<usize> {
        self.node_to_tensor_index_map.get(node_id).copied()
    }

    pub fn load_value(&mut self, node_id: usize, value: Value) -> Result<()> {
        match value.inner {
            InnerValue::Float32(data) => {
                let data = self
                    .instance_state
                    ._plan()
                    .device(0)
                    .expect("We always have one device")
                    .htod_float(data)?;
                let tensor = self
                    .get_tensor_from_node_id_mut(node_id)
                    .ok_or(Error::MissingData)?;
                tensor.init(data)
            }
            _ => unimplemented!(),
        }

        Ok(())
    }

    pub fn get_value(&self, node_id: usize) -> Result<Value> {
        let provider = self
            .instance_state
            ._plan()
            .device(0)
            .expect("We always have one device");

        let tensor = self
            .get_tensor_from_node_id(node_id)
            .ok_or(Error::MissingData)?;
        let ptr = tensor.data().take().ok_or(Error::MissingData)?;

        match tensor.dtype() {
            DataType::Float => Ok(provider.dtoh_float(ptr)?.into()),
            _ => unimplemented!(),
        }
    }

    pub fn graph(&self) -> &Arc<Graph<Definition>> {
        self.instance_state.graph()
    }

    /// Get the tensor value given a node ID.
    fn get_tensor_from_node_id(&self, node_id: usize) -> Option<&Tensor<T::Data>> {
        self.tensor.get(node_id)
    }

    /// Get the tensor value given a node ID.
    fn get_tensor_from_node_id_mut(&mut self, node_id: usize) -> Option<&mut Tensor<T::Data>> {
        self.tensor.get_mut(node_id)
    }

    /// Get the index of the actual value.
    fn get_inner_index(&self, value_index: usize) -> Option<usize> {
        self.op_tensors.get(value_index).copied()
    }
}
