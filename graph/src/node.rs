use crate::device::{Device, TensorTr};
use std::alloc::{Allocator, Global};
use std::ptr::NonNull;

pub type Link<F> = Option<NonNull<Node<F>>>;

pub struct Node<D: Device> {
    tensor: D::Tensor,
    op: D::Op,
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
    pub fn new(tensor: D::Tensor, op: D::Op, allocator: D::Allocator) -> Self {
        Self {
            tensor,
            op,
            inputs: Vec::new_in(allocator.clone()),
            outputs: Vec::new_in(allocator),
        }
    }
    pub fn tensor(&self) -> &D::Tensor {
        &self.tensor
    }

    pub fn op(&self) -> &D::Op {
        &self.op
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

    pub fn execute(&mut self) {
        self.tensor.execute(self.inputs.as_slice()).unwrap();
    }
}
