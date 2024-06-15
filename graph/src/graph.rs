use crate::device::Device;
use crate::node::Node;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug)]
pub enum GraphError {
    InvalidTensor,
    TensorNotFound,
    LoopDetected,
}

pub struct Graph<D: Device> {
    nodes: Vec<Node<D>, D::Allocator>,
    inputs: Vec<usize, D::Allocator>,
    outputs: Vec<usize, D::Allocator>,
}

impl<D> Graph<D>
where
    D: Device,
{
    pub fn forward(&mut self) -> Result<()> {
        todo!()
    }
}
