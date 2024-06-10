use std::ptr::NonNull;
use crate::device::{Device, Tensor};

pub type Result<T> = std::result::Result<T, ()>;
pub type Link<F> = Option<NonNull<Node<F>>>;

pub struct Node<D: Device> {
    /// The tensor.
    tensor: D::Tensor,
    // Order from left to right.
    // Input nodes where self is the op and output.
    inputs: Vec<NonNull<Node<D>>, D::Allocator>,
    // Order from left to right.
    // Output node where self is the input.
    outputs: Vec<NonNull<Node<D>>, D::Allocator>,
}

impl<D> Node<D>
where
    D: Device,
{
    pub fn new(tensor: D::Tensor, allocator: D::Allocator) -> Self {
        Self {
            tensor,
            inputs: Vec::new_in(allocator.clone()),
            outputs: Vec::new_in(allocator),
        }
    }
    pub fn tensor(&self) -> &D::Tensor {
        &self.tensor
    }

    pub fn op(&self) -> Option<<D::Tensor as Tensor<D>>::Op> {
        self.tensor.op()
    }

    pub fn inputs(&self) -> &[NonNull<Node<D>>] {
        self.inputs.as_slice()
    }

    pub fn outputs(&self) -> &[NonNull<Node<D>>] {
        self.outputs.as_slice()
    }

    pub fn inputs_mut(&mut self) -> &mut Vec<NonNull<Node<D>>, D::Allocator> {
        self.inputs.as_mut()
    }

    pub fn outputs_mut(&mut self) -> &mut Vec<NonNull<Node<D>>, D::Allocator> {
        self.outputs.as_mut()
    }

    pub unsafe fn input_link(&mut self, node: NonNull<Node<D>>) {
        self.inputs.push(node);
    }

    pub unsafe fn output_link(&mut self, node: NonNull<Node<D>>) {
        self.outputs.push(node);
    }

    pub fn execute(&mut self) -> Result<()> {
        self.tensor.forward(self.inputs.as_slice()).unwrap();
        Ok(())
    }
}
