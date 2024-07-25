use crate::execution_state::ExecutionState;
use crate::tensor::Tensor;
use crate::Error;
use rmlk_ir::Attribute;
use std::collections::HashMap;

pub trait Kernel {
    type Data;

    fn compute(&self, ctx: &mut Context<Self::Data>) -> crate::Result<()>;
}

pub struct Context<'a, T> {
    execution_state: &'a mut ExecutionState<T>,
    input_start_index: usize,
    max_tensors: usize,
    output_start_index: usize,
}

impl<'a, T> Context<'a, T> {
    pub fn new(
        execution_state: &'a mut ExecutionState<T>,
        node_index: usize,
    ) -> crate::Result<Self> {
        let input_count = execution_state
            .get_input_count(node_index)
            .ok_or(Error::MissingNodeInGraph)?;

        let output_count = execution_state
            .get_output_count(node_index)
            .ok_or(Error::MissingNodeInGraph)?;

        Ok(Self {
            execution_state,
            input_start_index: node_index,
            max_tensors: input_count + output_count,
            output_start_index: node_index + input_count,
        })
    }

    pub fn get_input(&self, index: usize) -> crate::Result<&Tensor<T>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::NoTensorFound);
        }

        self.execution_state
            .get_tensor(node_index)
            .ok_or(Error::MissingTensor)
    }

    pub fn get_input_mut(&mut self, index: usize) -> crate::Result<&mut Tensor<T>> {
        let node_index = self.input_start_index + index;
        if self.output_start_index <= node_index {
            return Err(Error::NoTensorFound);
        }

        self.execution_state
            .get_tensor_mut(node_index)
            .ok_or(Error::MissingTensor)
    }

    pub fn get_output(&self, index: usize) -> crate::Result<&Tensor<T>> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_tensors < node_index {
            return Err(Error::NoTensorFound);
        }

        self.execution_state
            .get_tensor(node_index)
            .ok_or(Error::MissingTensor)
    }

    pub fn get_output_mut(&mut self, index: usize) -> crate::Result<&mut Tensor<T>> {
        let node_index = self.output_start_index + index;
        if self.input_start_index + self.max_tensors < node_index {
            return Err(Error::NoTensorFound);
        }

        self.execution_state
            .get_tensor_mut(node_index)
            .ok_or(Error::MissingTensor)
    }

    pub fn get_attributes(&self) -> Option<&HashMap<Box<str>, Attribute>> {
        Some(
            self.execution_state
                .get_node(self.input_start_index)?
                .attrs(),
        )
    }
}

#[derive(Clone)]
pub struct Allocator;

impl Allocator {
    pub fn alloc_with_value<T: Clone>(&self, value: T, size: usize) -> Box<[T]> {
        vec![value; size].into_boxed_slice()
    }
}
