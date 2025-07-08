use crate::core::device_service::{DeviceService, ValueStore};
use crate::core::execution_state::ExecutionState;
use anyhow::Result;
use rmlk_graph::Node;
use rmlk_schema::{Attribute, Definition};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

#[derive(Debug)]
pub enum ContextError {
    TensorIndexNotFound { node_id: usize },
    InvalidTensorIndex { index: usize },
    TensorNotFound { index: usize },
}

impl Display for ContextError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextError::TensorIndexNotFound { node_id } => {
                write!(f, "Tensor index not found: {}", node_id)
            }
            ContextError::InvalidTensorIndex { index } => {
                write!(f, "Invalid tensor index: {}", index)
            }
            ContextError::TensorNotFound { index } => write!(f, "Tensor not found: {}", index),
        }
    }
}

impl std::error::Error for ContextError {}

/// Computation context.
///
/// Context provides a simple API for kernel functions
/// that need access to the inputs, outputs and attributes
/// needed for performing the computation.
///
/// It's essentially a wrapper over [`ExecutionState`] that
/// provides safe and correct access to the values for the computation.
pub struct Context<'a, D: DeviceService> {
    /// State for executing the model.
    execution_state: &'a mut ExecutionState<D>,
    /// Max number of values for this computation
    /// including both inputs and outputs.
    max_values: usize,
    /// Index for finding the start of the sequence of input
    /// values for the computation.
    input_start_index: usize,
    /// Index for finding the start of the sequence of output
    /// values for the computation.
    output_start_index: usize,
    /// Node ID of the computation in the graph.
    original_node_id: usize,
}

impl<'a, D> Context<'a, D>
where
    D: DeviceService,
{
    pub fn new(execution_state: &'a mut ExecutionState<D>, node_id: usize) -> Result<Self> {
        let node_index = execution_state
            .get_tensor_index(&node_id)
            .ok_or_else(|| ContextError::TensorIndexNotFound { node_id })?;
        let input_count = execution_state
            .graph()
            .get_node(node_id)
            .expect("the runtime to pass a node ID that is consistent with the execution state")
            .inputs()
            .len();
        let output_count = execution_state
            .graph()
            .get_node(node_id)
            .expect("the runtime to pass a node ID that is consistent with the execution state")
            .outputs()
            .len();

        Ok(Self {
            execution_state,
            input_start_index: node_index,
            max_values: input_count + output_count,
            output_start_index: node_index + input_count,
            original_node_id: node_id,
        })
    }

    pub fn get_input(&self, index: usize) -> Result<<D::Store as ValueStore>::Value> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(ContextError::InvalidTensorIndex { index: node_index }.into());
        }

        self.execution_state
            .get_tensor(node_index)
            .ok_or_else(|| ContextError::TensorNotFound { index: node_index })
            .map_err(Into::into)
    }

    pub fn input_exists(&self, index: usize) -> bool {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return false;
        }

        self.execution_state.get_tensor(node_index).is_some()
    }

    pub fn get_output(&self, index: usize) -> Result<<D::Store as ValueStore>::Value> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_values < node_index {
            return Err(ContextError::InvalidTensorIndex { index: node_index }.into());
        }

        self.execution_state
            .get_tensor(node_index)
            .ok_or_else(|| ContextError::TensorNotFound { index: node_index })
            .map_err(Into::into)
    }

    pub fn get_node(&self) -> Option<&Node<Definition>> {
        self.execution_state.graph().get_node(self.original_node_id)
    }

    pub fn get_output_node(&self) -> Option<&Node<Definition>> {
        let op_node = self.get_node()?;
        let output = op_node.outputs().iter().next()?;
        self.execution_state.graph().get_node(*output)
    }

    pub fn get_attributes(&self) -> Option<Rc<HashMap<Box<str>, Attribute>>> {
        self.execution_state
            .graph()
            .get_node(self.original_node_id)?
            .value()
            .attrs()
    }

    pub fn execution_state(&self) -> &ExecutionState<D> {
        self.execution_state
    }

    pub fn execution_state_mut(&mut self) -> &mut ExecutionState<D> {
        self.execution_state
    }
}
