use std::sync::mpsc;
use rmlk_tensor::Op;
use rmlk_tensor::provider::Provider;
use crate::node::Node;
use crate::provider::Provider;
use crate::tensor::Tensor;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug)]
pub enum GraphError {
    InvalidTensor,
    TensorNotFound,
}

pub struct Graph<P, T> {
    nodes: Vec<Node<T>>,
    leaves: Vec<Node<T>>,
    provider: Provider,
}

impl<P, T> Graph<P, T>
where
    T: Tensor,
{
    pub fn compute(&self) -> Result<()> {
        for node in &self.nodes {
            self.provider.compute(node.op(), node.src());
        }

        Ok(())
    }

}