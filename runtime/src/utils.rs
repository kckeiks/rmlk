use num_traits::Num;
use std::ops::AddAssign;

pub fn calculate_stride<T: Num + Copy + AddAssign>(shape: &[T], stride: &mut [T]) {
    // Clear it before using it.
    stride.fill(T::zero());

    let ndims = shape.len();

    debug_assert_eq!(ndims, stride.len());

    stride[ndims - 1] = T::one();
    for i in (0..ndims - 1).rev() {
        stride[i] += stride[i + 1] * shape[i + 1];
    }
}
