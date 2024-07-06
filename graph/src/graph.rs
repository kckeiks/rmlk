use crate::node::Node;

pub type Result<T> = std::result::Result<T, GraphError>;

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
    initializers: Vec<rmlk_ir::Tensor>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    nodes: Vec<Node>,
}

impl Graph {
    pub(crate) fn new(
        initializers: Vec<rmlk_ir::Tensor>,
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

    pub fn inputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.inputs.iter().copied()
    }

    pub fn outputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outputs.iter().copied()
    }

    pub fn initializers(&self) -> impl Iterator<Item = &rmlk_ir::Tensor> + '_ {
        self.initializers.iter()
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_initial_tensor(&self, id: usize) -> Option<&rmlk_ir::Tensor> {
        self.initializers.get(id)
    }
}
