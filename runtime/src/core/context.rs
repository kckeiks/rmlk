use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::tensor::Tensor;
use crate::core::DeviceService;
use rmlk_ir::Attribute;
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
    pub fn new(execution_state: &'a mut ExecutionState<D>, index: usize) -> Result<Self> {
        let node_index = execution_state
            .get_value_index(&index)
            .ok_or(Error::MissingData)
            .unwrap();

        let input_count = execution_state
            .get_input_count(index)
            .ok_or(Error::ContextError)?;

        let output_count = execution_state
            .get_output_count(index)
            .ok_or(Error::ContextError)?;

        Ok(Self {
            execution_state,
            input_start_index: node_index,
            max_values: input_count + output_count,
            output_start_index: node_index + input_count,
            original_node_id: index,
        })
    }

    pub fn get_input(&self, index: usize) -> Result<&Tensor<D::Data>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::ContextError);
        }

        self.execution_state
            .get_value(node_index)
            .ok_or(Error::ContextError)
    }

    pub fn get_input_mut(&mut self, index: usize) -> Result<&mut Tensor<D::Data>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::ContextError);
        }

        self.execution_state
            .get_value_mut(node_index)
            .ok_or(Error::ContextError)
    }

    #[cfg(test)]
    pub fn get_output(&self, index: usize) -> Result<&Tensor<D::Data>> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_values < node_index {
            return Err(Error::ContextError);
        }

        self.execution_state
            .get_value(node_index)
            .ok_or(Error::ContextError)
    }

    pub fn get_output_mut(&mut self, index: usize) -> Result<&mut Tensor<D::Data>> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_values < node_index {
            return Err(Error::ContextError);
        }

        self.execution_state
            .get_value_mut(node_index)
            .ok_or(Error::ContextError)
    }

    pub fn get_attributes(&self) -> Option<&HashMap<Box<str>, Attribute>> {
        self.execution_state
            .get_node(self.original_node_id)?
            .def()
            .attrs()
    }
}
