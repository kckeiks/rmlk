use crate::core::allocators::ScratchAllocator;
use crate::core::device_service::DeviceService;
use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::core::instance_state::ModelInstanceState;
use crate::core::store::TensorStore;
use crate::core::tensor::Tensor;
use crate::core::tensor_handle::{DstTensorId, SrcTensorId};
use crate::core::value::{InnerValue, Value};
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
    tensor_store: TensorStore<T::Data>,
    /// Tensor indices for finding an operation's tensor values.
    ///
    /// The order is inputs, optional inputs and outputs.
    op_tensors: Box<[usize]>,
    /// Maps a computation given by a node ID
    /// to the start of the node's values in `values`.
    node_to_tensor_index_map: HashMap<usize, usize>,
    /// Reference to the model instance state.
    instance_state: Arc<ModelInstanceState<T>>,
    /// Scratch buffer allocator.
    scratch_alloc: ScratchAllocator,
}

impl<T> ExecutionState<T>
where
    T: DeviceService,
{
    pub fn new(
        instance_state: Arc<ModelInstanceState<T>>,
        store: TensorStore<T::Data>,
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
                debug_assert!(store.get(*input).is_some());

                if graph.get_node(*input).is_some() {
                    node_values.push(*input);
                } else {
                    return Err(InternalError::ExecutionState(format!(
                        "the input `{}` for node `{node_id}` does not exist in the graph",
                        *input
                    )));
                }
            }

            for output in node.outputs() {
                debug_assert!(store.get(*output).is_some());

                if graph.get_node(*output).is_some() {
                    node_values.push(*output);
                } else {
                    return Err(InternalError::ExecutionState(format!(
                        "the output `{}` for node `{node_id}` does not exist in the graph",
                        *output
                    )));
                }
            }
        }

        trace!("node_values={:?}", node_values);
        trace!("node_to_value_index_map={:?}", node_to_value_index_map);

        Ok(Self {
            tensor_store: store,
            node_to_tensor_index_map: node_to_value_index_map,
            op_tensors: node_values.into_boxed_slice(),
            instance_state,
            scratch_alloc: ScratchAllocator::new(),
        })
    }

    /// Get a reference to the node.
    pub fn get_node(&self, node_id: usize) -> Option<&Node<Definition>> {
        self.instance_state.graph().get_node(node_id)
    }

    /// Get a tensor value.
    ///
    /// The value index for a given computation can be
    /// found using [`ExecutionState::get_tensor_index`].
    pub fn get_tensor(&self, value_index: usize) -> Option<Tensor<T::Data>> {
        let index = self.get_inner_index(value_index)?;
        self.tensor_store.get(index)
    }

    /// Get the starting index for the values of a node.
    pub fn get_tensor_index(&self, node_id: &usize) -> Option<usize> {
        self.node_to_tensor_index_map.get(node_id).copied()
    }

    /// Load the value for the given node into the device.
    /// Returns an error if the node does not have a corresponding value,
    /// like for instance, a node that corresponds to an operation.
    pub fn load_value(&mut self, node_id: usize, value: Value) -> Result<()> {
        // Todo: should we also return an error when a user tries to update a constant?
        match value.inner {
            InnerValue::Float32(data) => {
                let data = self
                    .instance_state
                    ._plan()
                    .device(0)
                    .expect("We always have one device")
                    .htod_float(data)?;
                let mut tensor = self.get_tensor_from_node_id(node_id).ok_or_else(|| {
                    InternalError::ExecutionState(format!(
                        "failed to load value: missing tensor for node {node_id}"
                    ))
                })?;
                tensor.set_dev_data(data);
            }
            _ => unimplemented!(),
        }

        Ok(())
    }

    /// Gets a copy of the value from the device for the given node.
    /// Returns an error if the node does not have a corresponding value,
    /// like for instance, a node that corresponds to an operation.
    pub fn get_value(&self, node_id: usize) -> Result<Value> {
        let provider = self
            .instance_state
            ._plan()
            .device(0)
            .expect("We always have one device");

        let tensor = self.get_tensor_from_node_id(node_id).ok_or_else(|| {
            InternalError::ExecutionState(format!(
                "failed to get value: missing tensor for node {node_id}"
            ))
        })?;

        let ptr = tensor.dev_data_ptr().take().ok_or_else(|| {
            InternalError::ExecutionState(format!(
                "failed to get value: empty tensor for node {node_id}"
            ))
        })?;

        match tensor.dtype() {
            DataType::Float => Ok(provider.dtoh_float(&ptr)?.into()),
            _ => unimplemented!(),
        }
    }

    /// Get a shared reference to the computational graph of the model.
    pub fn graph(&self) -> &Arc<Graph<Definition>> {
        self.instance_state.graph()
    }

    /// Get a read-only reference to the scratch allocator.
    pub fn scratch_alloc(&self) -> &ScratchAllocator {
        &self.scratch_alloc
    }

    /// Get a mutable reference to the scratch allocator.
    pub fn scratch_alloc_mut(&mut self) -> &mut ScratchAllocator {
        &mut self.scratch_alloc
    }

    /// Get a read-only reference to the scratch allocator.
    pub fn scratch_alloc_clone(&self) -> ScratchAllocator {
        self.scratch_alloc.clone()
    }

    /// Copies the shape data from the source's shape buffer.
    pub fn copy_shape_from_within(&mut self, src: SrcTensorId, dst: DstTensorId) -> Result<()> {
        self.tensor_store
            .copy_shape_from_within(src.into(), dst.into())
    }

    /// Copies the shape data from the src slice.
    pub fn copy_shape_from_slice(&mut self, src: &[usize], dst: DstTensorId) -> Result<()> {
        self.tensor_store.copy_shape_from_slice(src, dst.into())
    }

    /// Get the tensor value given a node ID.
    fn get_tensor_from_node_id(&self, node_id: usize) -> Option<Tensor<T::Data>> {
        self.tensor_store.get(node_id)
    }

    /// Get the index of the actual value.
    fn get_inner_index(&self, value_index: usize) -> Option<usize> {
        self.op_tensors.get(value_index).copied()
    }
}
