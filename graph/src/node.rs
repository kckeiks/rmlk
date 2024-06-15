use crate::device::{Device, Tensor};
use crate::op::Op;

pub type Result<T> = std::result::Result<T, NodeError>;

pub enum NodeError {
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

    pub fn tensor(&self) -> &D::Tensor {
        &self.tensor
    }

    pub fn op(&self) -> Op {
        self.op
    }

    pub fn inputs(&self) -> &[usize] {
        self.inputs.as_slice()
    }

    pub fn inputs_mut(&mut self) -> &mut Vec<Node<D>, D::Allocator> {
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

    pub fn execute(&mut self, inputs: &[Node<D>]) -> Result<()> {
        self.tensor
            .as_mut()
            .ok_or(NodeError::Unknown)?
            .forward(inputs)
            .map_err(|_| NodeError::Unknown)
    }
}
