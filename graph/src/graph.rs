use std::alloc::Allocator;
use crate::device::Provider;
use crate::node::{Node};
use std::sync::Arc;

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
    initializers: Vec<rmlk_hir::Tensor, A>,
    inputs: Vec<usize, A>,
    nodes: Vec<Node<A>, A>,
    outputs: Vec<usize, A>,
}

impl<A> Graph<A>
where
    A: Allocator + Clone,
{
    pub(crate) fn new(
        initializers: Vec<rmlk_hir::Tensor, A>,
        inputs: Vec<usize, A>,
        nodes: Vec<Node<A>, A>,
        outputs: Vec<usize, A>,
    ) -> Self {
        Self {
            initializers,
            inputs,
            nodes,
            outputs,
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Arc<Node<A>, A>> + '_ {
        self.nodes.iter()
    }

    pub fn inputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outputs.iter().copied()
    }

    pub fn outputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outputs.iter().copied()
    }

    pub fn get_node(&self, id: usize) -> Option<&Node<A>> {
        self.nodes.get(id)
    }
}

