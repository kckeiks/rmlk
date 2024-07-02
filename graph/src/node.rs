use rmlk_schema::{DataType, Op};
use std::alloc::Allocator;

pub type Result<T> = std::result::Result<T, NodeError>;

#[derive(Debug)]
pub enum NodeError {
    ExecuteNoOpAttempt,
    Unknown,
}

pub struct Node<A: Allocator> {
    /// The node's Provider.
    ///
    /// Each node is assigned to a single Provider.
    provider: Option<u32>,
    /// The node's operation.
    ///
    /// If the op is NoOp, this node is an input and graph leaf.
    op: Op,
    /// Inputs for this node.
    ///
    /// An ID may correspond to a node or
    /// an initial tensor.
    inputs: Vec<usize, A>,
    /// Outputs for this node.
    ///
    /// An ID may correspond to a node or
    /// an initial tensor.
    outputs: Vec<usize, A>,
    /// The node's definition.
    definition: Definition<A>,
}

impl<A> Node<A>
where
    A: Allocator + Clone,
{
    pub fn new(
        op: Op,
        inputs: Vec<usize, A>,
        outputs: Vec<usize, A>,
        definition: Definition<A>,
    ) -> Self {
        Self {
            op,
            provider: None,
            inputs,
            outputs,
            definition,
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

/// The definition for this node's inputs and outputs.
pub struct Definition<A: Allocator> {
    pub shape: Vec<usize, A>,
    pub dtype: DataType,
}
