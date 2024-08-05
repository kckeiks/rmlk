use num_traits::Num;
use std::ops::AddAssign;

pub fn calculate_stride<T: Num + Copy + AddAssign>(shape: &[T], stride: &mut [T]) {
    let dims = shape.len();

    debug_assert_eq!(dims, stride.len());

    stride[dims - 1] = T::one();
    for i in (0..dims - 1).rev() {
        stride[i] += stride[i + 1] * shape[i + 1];
    }
}
