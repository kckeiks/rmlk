use crate::node::Node;

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

pub struct Graph<T> {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    nodes: Vec<Node<T>>,
}

impl<T> Graph<T> {
    pub fn new(inputs: Vec<usize>, nodes: Vec<Node<T>>, outputs: Vec<usize>) -> Self {
        Self {
            inputs,
            outputs,
            nodes,
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node<T>> + '_ {
        self.nodes.iter()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_iter(&self) -> impl Iterator<Item = (usize, &Node<T>)> {
        self.nodes.iter().enumerate()
    }

    pub fn get_node(&self, id: usize) -> Option<&Node<T>> {
        self.nodes.get(id)
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
}
