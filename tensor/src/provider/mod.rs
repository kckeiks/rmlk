use crate::tensor::Tensor;
use rmlk_hir::DataType;
use std::alloc::Allocator;

pub mod cuda;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Unknown,
}

pub trait Provider: Sized {
    type Data;
    type Allocator: Allocator + Clone;
    fn allocator(&self) -> Self::Allocator;
    fn tensor(&self, input: rmlk_hir::Tensor) -> Result<Tensor<Self>>;
    fn tensor_from_dtype_with_shape(
        &self,
        data_type: DataType,
        shape: Vec<usize, Self::Allocator>,
    ) -> Result<Tensor<Self>>;
    fn tensor_from_dtype(&self, data_type: DataType) -> Result<Tensor<Self>>;
}
