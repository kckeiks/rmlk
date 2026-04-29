use crate::providers::cuda::Tensor;
use crate::utils::{SCALAR_SHAPE, SCALAR_STRIDE};
use std::cell::Ref;

pub fn get_kernel_safe_shape_and_stride(tensor: &Tensor) -> (Ref<'_, [usize]>, Ref<'_, [usize]>) {
    if tensor.is_scalar() {
        scalar_shape_and_stride(tensor)
    } else {
        (tensor.shape(), tensor.stride())
    }
}

pub fn scalar_shape_and_stride(tensor: &Tensor) -> (Ref<'_, [usize]>, Ref<'_, [usize]>) {
    (
        Ref::map(tensor.shape(), |_| SCALAR_SHAPE),
        Ref::map(tensor.shape(), |_| SCALAR_STRIDE),
    )
}
