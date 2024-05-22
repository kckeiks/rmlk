use rmlk_tensor::device::Device;
use crate::node::Node;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug)]
pub enum GraphError {
    InvalidTensor,
    TensorNotFound,
}

pub struct Graph<D: Device> {
    nodes: Vec<Node<D>>,
    leaves: Vec<Node<D>>,

}

// impl<D> Graph<D>
// where
//     D: Device,
// {
//
// }
