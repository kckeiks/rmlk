use crate::tensor::dtype::DType;
use crate::tensor::op::Op;
use crate::tensor::raw::{RawTensorPtr, MAX_DIMS, MAX_SRC};
use std::alloc::Layout;
use std::ptr::NonNull;

pub struct Tensor<A> {
    pub dtype: DType,
    pub shape: [usize; MAX_DIMS],
    pub stride: [usize; MAX_DIMS],
    pub alloc: A,
    pub layout: Layout,
    pub data: Option<NonNull<[u8]>>,
}
