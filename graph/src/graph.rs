use crate::node::Node;
use std::collections::HashMap;

#[derive(Debug)]
pub enum GraphError {
    InvalidTensor,
    TensorNotFound,
    UnknownNode,
    LoopDetected,
    Unknown,
    NodeInUse,
    ComputationFailed,
}

pub struct Graph {
    initializers: HashMap<usize, rmlk_schema::Tensor>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new(
        initializers: HashMap<usize, rmlk_schema::Tensor>,
        inputs: Vec<usize>,
        nodes: Vec<Node>,
        outputs: Vec<usize>,
    ) -> Self {
        Self {
            initializers,
            inputs,
            outputs,
            nodes,
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> + '_ {
        self.nodes.iter()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes_slice(&self) -> &[Node] {
        self.nodes.as_slice()
    }

    pub fn inputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.inputs.iter().copied()
    }

    pub fn outputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outputs.iter().copied()
    }

    pub fn outputs_slice(&self) -> &[usize] {
        self.outputs.as_slice()
    }

    pub fn initializers(&self) -> impl Iterator<Item = (&usize, &rmlk_schema::Tensor)> + '_ {
        self.initializers.iter()
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_initial_tensor(&self, id: usize) -> Option<&rmlk_schema::Tensor> {
        self.initializers.get(&id)
    }
}
