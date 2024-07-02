use crate::node::Node;
use std::alloc::Allocator;

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

pub struct Graph<A: Allocator> {
    initializers: Vec<rmlk_ir::Tensor, A>,
    inputs: Vec<usize, A>,
    outputs: Vec<usize, A>,
    nodes: Vec<Node<A>, A>,
}

impl<A> Graph<A>
where
    A: Allocator + Clone,
{
    pub(crate) fn new(
        initializers: Vec<rmlk_ir::Tensor, A>,
        inputs: Vec<usize, A>,
        nodes: Vec<Node<A>, A>,
        outputs: Vec<usize, A>,
    ) -> Self {
        Self {
            initializers,
            inputs,
            outputs,
            nodes,
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node<A>> + '_ {
        self.nodes.iter()
    }

    pub fn inputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outputs.iter().copied()
    }

    pub fn outputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outputs.iter().copied()
    }

    pub fn initializers(&self) -> impl Iterator<Item = &rmlk_ir::Tensor> + '_ {
        self.initializers.iter()
    }

    pub fn get_node(&self, id: usize) -> Option<&Node<A>> {
        self.nodes.get(id)
    }

    pub fn get_initial_tensor(&self, id: usize) -> Option<&rmlk_ir::Tensor> {
        self.initializers.get(id)
    }
}
