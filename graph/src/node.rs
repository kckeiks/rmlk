use crate::device::{Device, Tensor};
use crate::op::Op;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, NodeError>;

#[derive(Debug)]
pub enum NodeError {
    ExecuteNoOpAttempt,
    MissingTensor,
    Unknown,
}

pub struct Node<D: Device> {
    /// The device.
    device: D,
    /// The identifier of the operator for this tensor.
    op: Op,
    /// The tensor.
    tensor: Option<D::Tensor>,
    // Order from left to right.
    // Input nodes where self is the op and output.
    inputs: Vec<usize, D::Allocator>,
    // Order from left to right.
    // Output node where self is the input.
    outputs: Vec<usize, D::Allocator>,
}

impl<D> Node<D>
where
    D: Device,
{
    pub fn new(op: Op, device: D) -> Self {
        Self {
            op,
            tensor: None,
            inputs: Vec::new_in(device.allocator()),
            outputs: Vec::new_in(device.allocator()),
            device,
        }
    }

    pub fn new_with_tensor(op: Op, device: D, tensor: D::Tensor) -> Self {
        Self {
            op,
            tensor: Some(tensor),
            inputs: Vec::new_in(device.allocator()),
            outputs: Vec::new_in(device.allocator()),
            device,
        }
    }

    pub fn tensor(&self) -> Option<&D::Tensor> {
        self.tensor.as_ref()
    }

    pub fn set_tensor(&mut self, tensor: D::Tensor) {
        self.tensor = Some(tensor);
    }

    pub fn op(&self) -> Op {
        self.op
    }

    pub fn inputs(&self) -> &[usize] {
        self.inputs.as_slice()
    }

    pub fn inputs_mut(&mut self) -> &mut Vec<usize, D::Allocator> {
        self.inputs.as_mut()
    }

    pub fn set_input(&mut self, node_id: usize) {
        self.inputs.push(node_id);
    }

    pub fn outputs(&self) -> &[usize] {
        self.outputs.as_slice()
    }

    pub fn set_output(&mut self, node_id: usize) {
        self.outputs.push(node_id);
    }

    pub fn execute(
        &mut self,
        inputs: impl Iterator<Item = Arc<Node<D>, D::Allocator>>,
    ) -> Result<()> {
        if matches!(self.op, Op::NoOp) {
            return Err(NodeError::ExecuteNoOpAttempt);
        }

        self.tensor
            .as_mut()
            .ok_or(NodeError::MissingTensor)?
            .compute(self.op, inputs)
            .map_err(|_| NodeError::Unknown)
    }
}
