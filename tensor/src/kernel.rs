use crate::execution_state::ExecutionState;
use crate::tensor::Tensor;
use crate::Error;

pub trait Kernel {
    type Data;

    fn compute(&self, ctx: &mut Context<Self::Data>) -> crate::Result<()>;
}

pub struct Context<T> {
    execution_state: ExecutionState<T>,
    current_node: usize,
    input_count: usize,
}

impl<T> Context<T> {
    pub fn new(execution_state: ExecutionState<T>, node_id: usize) -> crate::Result<Self> {
        let input_count = execution_state
            .get_input_count(node_id)
            .ok_or(Error::MissingNodeInGraph)?;
        Ok(Self {
            execution_state,
            current_node: node_id,
            input_count,
        })
    }

    pub fn get_input(&self, index: usize) -> crate::Result<&Tensor<T>> {
        self.execution_state
            .get_tensor(self.current_node, index)
            .ok_or(Error::MissingTensor)
    }

    pub fn get_output(&mut self, index: usize) -> crate::Result<&Tensor<T>> {
        self.execution_state
            .get_tensor(
                self.current_node,
                self.input_count.checked_add(index).ok_or(Error::Overflow)?,
            )
            .ok_or(Error::MissingTensor)
    }

    pub fn get_output_mut(&mut self, index: usize) -> crate::Result<&mut Tensor<T>> {
        self.execution_state
            .get_tensor_mut(
                self.current_node,
                self.input_count.checked_add(index).ok_or(Error::Overflow)?,
            )
            .ok_or(Error::MissingTensor)
    }
}
