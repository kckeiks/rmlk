pub mod cpu;

use std::alloc::Allocator;
use std::ptr::NonNull;
use crate::node::Node;

#[derive(Debug)]
pub enum DeviceError {
    Unknown,
}

pub type Result<T> = std::result::Result<T, DeviceError>;

pub trait Device: Clone {
    type Tensor: Tensor<Self>;
    type Allocator: Allocator + Clone;
    fn allocator(&self) -> Self::Allocator;
    fn new_tensor(&self, input: EncodedTensor) -> Result<Self::Tensor>;
}

pub struct EncodedTensor {
    pub op: bool,
    pub value: u32,
}

pub trait Tensor<D: Device> {
    type Op;
    fn op(&self) -> Option<Self::Op>;
    fn forward(&mut self, inputs: &[NonNull<Node<D>>]) -> Result<()>;
}
