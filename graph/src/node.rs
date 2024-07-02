use rmlk_ir::{DataType, Op};

pub type Result<T> = std::result::Result<T, NodeError>;

#[derive(Debug)]
pub enum NodeError {
    ExecuteNoOpAttempt,
    Unknown,
}

pub struct Node {
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
    inputs: Vec<usize>,
    /// Outputs for this node.
    ///
    /// An ID may correspond to a node or
    /// an initial tensor.
    outputs: Vec<usize>,
    /// The node's definition.
    definition: Definition,
}

impl Node {
    pub fn new(op: Op, inputs: Vec<usize>, outputs: Vec<usize>, definition: Definition) -> Self {
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
pub struct Definition {
    pub shape: Vec<usize>,
    pub dtype: DataType,
}
