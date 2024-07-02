use crate::provider::kernel::OpKernel;
use crate::provider::Result;
use crate::tensor::Tensor;
use rmlk_ir::{DataType, Op};

/// The execution provider performs kernel execution
pub trait Provider: Sized {
    type Data;
    /// Allocates a tensor.
    fn allocate(&mut self, dtype: DataType, shape: Vec<usize>) -> Result<&mut Tensor<Self>>;
    fn get_tensor(&self, dtype: DataType, id: usize) -> Result<&Tensor<Self>>;
    fn get_tensor_mut(&mut self, dtype: DataType, id: usize) -> Result<&mut Tensor<Self>>;
}
