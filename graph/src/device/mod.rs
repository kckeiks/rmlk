mod cpu;

use crate::node::Node;
pub use cpu::{CpuDevice, CpuTensor};
use std::alloc::Allocator;
use std::ptr::NonNull;

pub type Result<T> = std::result::Result<T, DeviceError>;

#[derive(Debug)]
pub enum DeviceError {
    Unknown,
}

pub trait Device: Clone {
    type Tensor: Tensor<Self>;
    type Allocator: Allocator + Clone;
    fn allocator(&self) -> Self::Allocator;
    fn tensor(&self, input: rmlk_hir::Tensor) -> Result<Self::Tensor>;
}

pub trait Tensor<D: Device> {
    fn forward(&mut self, inputs: &[Node<D>]) -> Result<()>;
}
