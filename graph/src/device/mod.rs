mod cpu;

use crate::node::Node;
use crate::Op;
pub use cpu::{CpuDevice, CpuTensor};
use std::alloc::Allocator;
use std::fmt::Debug;
use std::ptr::NonNull;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, DeviceError>;

#[derive(Debug)]
pub enum DeviceError {
    Unknown,
    InvalidOp,
    InvalidShape,
    InvalidStride,
    InvalidInputDataType,
    MissingInput,
}

pub trait Device: Clone {
    type Tensor: Tensor<Self> + Debug;
    type Allocator: Allocator + Clone;
    fn allocator(&self) -> Self::Allocator;
    fn tensor(&self) -> Self::Tensor;
    fn tensor_from_hir(&self, input: rmlk_hir::Tensor) -> Result<Self::Tensor>;
}

pub trait Tensor<D: Device> {
    fn compute<'a>(
        &mut self,
        op: Op,
        inputs: impl Iterator<Item = Arc<Node<D>, D::Allocator>>,
    ) -> Result<()>;
}
