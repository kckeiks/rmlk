use num_traits::Num;
use std::cmp;
use std::ops::AddAssign;

pub fn calculate_stride<T: Num + Copy + AddAssign>(shape: &[T], stride: &mut [T]) {
    let ndims = shape.len();

    debug_assert_eq!(ndims, stride.len());

    stride[ndims - 1] = T::one();
    for i in (0..ndims - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

pub fn broadcast_stride(a: &[usize], b: &[usize], b_stride: &mut [usize]) -> Result<bool, ()> {
 todo!()
}

pub fn broadcast(a: &[usize], b: &[usize], dst: &mut [usize]) -> bool {
    let ndims = dst.len();

    // Ensure neither input shape exceeds the destination dimensions.
    if a.len() > ndims || b.len() > ndims {
        return false;
    }

    // Compute offsets for aligning shorter arrays with the destination.
    let a_offset = ndims - a.len();
    let b_offset = ndims - b.len();

    for i in 0..ndims {
        // Fill missing dimensions with 1.
        let a_dim = a.get(i - a_offset).copied().unwrap_or(1);
        let b_dim = b.get(i - b_offset).copied().unwrap_or(1);

        // Check broadcasting rules.
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return false;
        }

        dst[i] = cmp::max(a_dim, b_dim);
    }

    true
}
