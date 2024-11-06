use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::tensor::Tensor;
use crate::core::DeviceService;
use rmlk_schema::Attribute;
use std::collections::HashMap;

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
            .ok_or_else(|| {
                Error::UnknownTensor(format!(
                    "failed to create a context: failed to find tensor for node {node_id}"
                ))
            })
            .unwrap();
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

    pub fn get_input(&self, index: usize) -> Result<&Tensor<D::Data>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::Internal(format!(
                "context error: invalid node index {node_index}"
            )));
        }

        self.execution_state.get_tensor(node_index).ok_or_else(|| {
            Error::UnknownTensor(format!(
                "context error: failed to find tensor for node {node_index}"
            ))
        })
    }

    pub fn get_input_mut(&mut self, index: usize) -> Result<&mut Tensor<D::Data>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::Internal(format!(
                "context error: invalid node index {node_index}"
            )));
        }

        self.execution_state
            .get_tensor_mut(node_index)
            .ok_or_else(|| {
                Error::UnknownTensor(format!(
                    "context error: failed to find tensor for node {node_index}"
                ))
            })
    }

    #[cfg(test)]
    pub fn get_output(&self, index: usize) -> Result<&Tensor<D::Data>> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_values < node_index {
            return Err(Error::Internal(format!(
                "context error: invalid node index {node_index}"
            )));
        }

        self.execution_state.get_tensor(node_index).ok_or_else(|| {
            Error::UnknownTensor(format!(
                "context error: failed to find tensor for node {node_index}"
            ))
        })
    }

    pub fn get_output_mut(&mut self, index: usize) -> Result<&mut Tensor<D::Data>> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_values < node_index {
            return Err(Error::Internal(format!(
                "context error: invalid node index {node_index}"
            )));
        }

        self.execution_state
            .get_tensor_mut(node_index)
            .ok_or_else(|| {
                Error::UnknownTensor(format!(
                    "context error: failed to find tensor for node {node_index}"
                ))
            })
    }

    pub fn get_attributes(&self) -> Option<&HashMap<Box<str>, Attribute>> {
        self.execution_state
            .graph()
            .get_node(self.original_node_id)?
            .value()
            .attrs()
    }
}
