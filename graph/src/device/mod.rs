mod cpu;

use crate::node::Node;
use crate::Op;
pub use cpu::{CpuDevice, CpuTensor};
use std::alloc::Allocator;
use std::fmt::Debug;
use std::sync::Arc;
use rmlk_hir::{DataType, TensorShape};

pub type Result<T> = std::result::Result<T, DeviceError>;

#[derive(Debug)]
pub enum DeviceError {
    Unknown,
    InvalidOp,
    InvalidShape,
    InvalidStride,
    InvalidInputDataType,
    MissingInput,
    MissingTensorForOp,
}

pub trait Device: Clone {
    type Tensor: Tensor<Self> + Debug;
    type Allocator: Allocator + Clone;
    fn allocator(&self) -> Self::Allocator;
    fn tensor(&self, input: rmlk_hir::Tensor) -> Result<Self::Tensor>;
    fn tensor_from_value(&self, data_type: DataType, shape: Vec<usize, Self::Allocator>) -> Result<Self::Tensor>;
}

pub trait Tensor<D: Device> {
    fn compute<'a>(
        &mut self,
        op: Op,
        inputs: impl Iterator<Item = Arc<Node<D>, D::Allocator>>,
    ) -> Result<()>;
}
