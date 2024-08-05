use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::tensor::Tensor;
use rmlk_ir::Attribute;
use std::collections::HashMap;

pub struct Context<'a, T> {
    execution_state: &'a mut ExecutionState<T>,
    input_start_index: usize,
    max_tensors: usize,
    output_start_index: usize,

    original_node_id: usize,
}

impl<'a, T> Context<'a, T> {
    pub fn new(
        execution_state: &'a mut ExecutionState<T>,
        node_tensor_map: &HashMap<usize, usize>,
        index: usize,
    ) -> Result<Self> {
        // println!("new context: {index}");
        let node_index = *node_tensor_map
            .get(&index)
            .ok_or(Error::MissingData)
            .unwrap();
        // println!("new node_index: {node_index}");

        let input_count = execution_state
            .get_input_count(index)
            .ok_or(Error::ContextError)?;

        let output_count = execution_state
            .get_output_count(index)
            .ok_or(Error::ContextError)?;

        Ok(Self {
            execution_state,
            input_start_index: node_index,
            max_tensors: input_count + output_count,
            output_start_index: node_index + input_count,
            original_node_id: index,
        })
    }

    pub fn get_input(&self, index: usize) -> Result<&Tensor<T>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::ContextError);
        }

        self.execution_state
            .get_tensor(node_index)
            .ok_or(Error::ContextError)
    }

    pub fn _get_input_mut(&mut self, index: usize) -> Result<&mut Tensor<T>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::ContextError);
        }

        self.execution_state
            .get_tensor_mut(node_index)
            .ok_or(Error::ContextError)
    }

    #[cfg(test)]
    pub fn get_output(&self, index: usize) -> Result<&Tensor<T>> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_tensors < node_index {
            return Err(Error::ContextError);
        }

        self.execution_state
            .get_tensor(node_index)
            .ok_or(Error::ContextError)
    }

    pub fn get_output_mut(&mut self, index: usize) -> Result<&mut Tensor<T>> {
        println!("index={index}");
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_tensors < node_index {
            return Err(Error::ContextError);
        }
        println!("node_index={node_index}");

        self.execution_state
            .get_tensor_mut(node_index)
            .ok_or(Error::ContextError)
    }

    pub fn get_attributes(&self) -> Option<&HashMap<Box<str>, Attribute>> {
        Some(
            self.execution_state
                .get_node(self.original_node_id)?
                .attrs(),
        )
    }
}
