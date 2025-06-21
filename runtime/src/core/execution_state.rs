use crate::core::allocators::ScratchAllocator;
use crate::core::device_service::DeviceService;
use crate::core::instance_state::ModelInstanceState;
use crate::core::plan::Plan;
use crate::core::store::TensorStore;
use crate::core::tensor::Tensor;
use crate::core::tensor_handle::{DstTensorId, SrcTensorId};
use crate::core::value::{InnerValue, Value};
use anyhow::Result;
use log::trace;
use rmlk_graph::{Graph, Node};
use rmlk_schema::Op;
use rmlk_schema::{DataType, Definition};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
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
            if matches!(node.value().op(), Op::NoOp) {
                continue;
            }

            let index = node_values.len();
            node_to_value_index_map.insert(node_id, index);

            for input in node.inputs() {
                debug_assert!(store.get(*input).is_some());

                if graph.get_node(*input).is_some() {
                    node_values.push(*input);
                } else {
                    return Err(ExecutionStateError::InputTensorNotFound { id: *input }.into());
                }
            }

            for output in node.outputs() {
                debug_assert!(store.get(*output).is_some());

                if graph.get_node(*output).is_some() {
                    node_values.push(*output);
                } else {
                    return Err(ExecutionStateError::OutputTensorNotFound { id: *output }.into());
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
            InnerValue::Float16(data) => {
                let data = self
                    .instance_state
                    ._plan()
                    .device(0)
                    .expect("We always have one device")
                    .htod_float16(data)?;
                let mut tensor = self
                    .get_tensor_from_node_id(node_id)
                    .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;
                tensor.set_dev_data(data);
            }
            InnerValue::Float32(data) => {
                let data = self
                    .instance_state
                    ._plan()
                    .device(0)
                    .expect("We always have one device")
                    .htod_float(data)?;
                let mut tensor = self
                    .get_tensor_from_node_id(node_id)
                    .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;
                tensor.set_dev_data(data);
            }
            InnerValue::Int32(data) => {
                let data = self
                    .instance_state
                    ._plan()
                    .device(0)
                    .expect("We always have one device")
                    .htod_i32(data)?;
                let mut tensor = self
                    .get_tensor_from_node_id(node_id)
                    .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;
                tensor.set_dev_data(data);
            }
            InnerValue::Int64(data) => {
                let data = self
                    .instance_state
                    ._plan()
                    .device(0)
                    .expect("We always have one device")
                    .htod_i64(data)?;
                let mut tensor = self
                    .get_tensor_from_node_id(node_id)
                    .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;
                tensor.set_dev_data(data);
            }
            InnerValue::Bool(data) => {
                let data = self
                    .instance_state
                    ._plan()
                    .device(0)
                    .expect("We always have one device")
                    .htod_bool(data)?;
                let mut tensor = self
                    .get_tensor_from_node_id(node_id)
                    .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;
                tensor.set_dev_data(data);
            }
        }

        if let Some(shape) = value.shape {
            self.compare_shapes(&shape, node_id)?;

            let tensor = self
                .get_tensor_from_node_id(node_id)
                .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;

            let dst_id = tensor.dst_id();
            self.copy_shape_from_slice(&shape, dst_id)?;
        }

        Ok(())
    }

    fn compare_shapes(&self, shape: &[usize], node_id: usize) -> Result<()> {
        let node = self
            .get_node(node_id)
            .expect("we already checked that it exists above");

        let shape_def = node
            .value()
            .shape()
            .ok_or(ExecutionStateError::MissingShape)?;

        if shape.len() != shape_def.len() {
            println!(
                "{} {:?} != {:?}",
                node.value().name().unwrap(),
                shape,
                shape_def
            );
            return Err(ExecutionStateError::RankMismatch.into());
        }

        for (idx, &dim) in shape.iter().enumerate() {
            if shape_def[idx] != dim {
                if shape_def[idx] != 0 || !(shape_def[idx] == 0 && node.value().has_dynamic_dims())
                {
                    return Err(ExecutionStateError::DimensionMismatch.into());
                }
            }
        }

        Ok(())
    }

    pub fn compare_value_and_def_shape(&self, node_id: usize) -> Result<()> {
        let tensor = self
            .get_tensor_from_node_id(node_id)
            .ok_or(ExecutionStateError::ComparisonFailed { id: node_id })?;
        self.compare_shapes(tensor.shape(), node_id)
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

        let tensor = self
            .get_tensor_from_node_id(node_id)
            .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;

        let ptr = tensor
            .dev_data_ptr()
            .take()
            .ok_or(ExecutionStateError::TensorNotFound { id: node_id })?;

        match tensor.dtype() {
            DataType::Float => Ok((provider.dtoh_float(&ptr)?, tensor.shape()).into()),
            DataType::Int32 => Ok((provider.dtoh_i32(&ptr)?, tensor.shape()).into()),
            DataType::Int64 => Ok((provider.dtoh_i64(&ptr)?, tensor.shape()).into()),
            DataType::Bool => Ok((provider.dtoh_bool(&ptr)?, tensor.shape()).into()),
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

#[derive(Debug)]
pub enum ExecutionStateError {
    InputTensorNotFound { id: usize },
    OutputTensorNotFound { id: usize },
    TensorNotFound { id: usize },
    ComparisonFailed { id: usize },
    MissingShape,
    RankMismatch,
    DimensionMismatch,
}

impl Display for ExecutionStateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionStateError::InputTensorNotFound { id } => {
                write!(f, "Input tensor not found: {}", id)
            }
            ExecutionStateError::OutputTensorNotFound { id } => {
                write!(f, "Output tensor not found: {}", id)
            }
            ExecutionStateError::TensorNotFound { id } => {
                write!(f, "Tensor not found: {}", id)
            }
            ExecutionStateError::MissingShape => {
                write!(f, "Tensor is missing shape")
            }
            ExecutionStateError::RankMismatch => {
                write!(
                    f,
                    "the rank of argument does not match the rank defined for the tensor"
                )
            }
            ExecutionStateError::DimensionMismatch => {
                write!(
                    f,
                    "given shape does not match the shape defined for the tensor"
                )
            }
            ExecutionStateError::ComparisonFailed { id } => {
                write!(f, "Comparison failed for node `{}`", id)
            }
        }
    }
}

impl Error for ExecutionStateError {}
