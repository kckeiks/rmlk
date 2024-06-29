use std::alloc::Allocator;
use crate::device::{Provider, Tensor};
use crate::op::Op;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, NodeError>;

#[derive(Debug)]
pub enum NodeError {
    ExecuteNoOpAttempt,
    Unknown,
}

pub struct Node<A: Allocator> {
    /// The identifier of the operator for this node.
    op: Op,
    // Order from left to right.
    // Input nodes where self is the op and output.
    inputs: Vec<usize, A>,
    // Order from left to right.
    // Output node where self is the input.
    outputs: Vec<usize, A>,
}

impl<A> Node<A>
where
    A: Allocator + Clone,
{
    pub fn new_with_alloc(op: Op, alloc: A) -> Self {
        Self::new(op, Vec::new_in(alloc.clone()), Vec::new_in(alloc))
    }

    pub fn new(op: Op, inputs: Vec<usize, A>, outputs: Vec<usize, A>) -> Self {
        Self {
            op,
            inputs,
            outputs,
        }
    }

    pub fn op(&self) -> Op {
        self.op
    }

    pub fn inputs(&self) -> &[usize] {
        self.inputs.as_slice()
    }

    pub fn add_input(&mut self, node_id: usize) {
        self.inputs.push(node_id);
    }

    pub fn outputs(&self) -> &[usize] {
        self.outputs.as_slice()
    }

    pub fn add_output(&mut self, node_id: usize) {
        self.outputs.push(node_id);
    }
}

