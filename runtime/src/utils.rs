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

pub fn compute_broadcast_stride(
    a_shape: &[usize],
    b_shape: &[usize],
    a_stride: &[usize],
    b_stride: &[usize],
    strides: &mut [usize],
) {
    assert!(a_shape.len() > 0 || b_shape.len() > 0);

    let mid = cmp::max(a_shape.len(), b_shape.len());

    assert_eq!(mid, strides.len() / 2);

    let (strides_a, strides_b) = strides.split_at_mut(mid);

    let a_len = a_shape.len();
    let b_len = b_shape.len();

    for i in (0..mid).rev() {
        if let Some(a_i) = i.checked_sub(mid - a_len) {
            if i.checked_sub(mid - b_len)
                .map(|idx| a_shape[a_i] == b_shape[idx] || b_shape[idx] == 1)
                .unwrap_or(false)
            {
                strides_a[i] = a_stride[a_i];
            }
        }

        if let Some(b_i) = i.checked_sub(mid - b_len) {
            if i.checked_sub(mid - a_len)
                .map(|idx| a_shape[idx] == b_shape[b_i] || a_shape[idx] == 1)
                .unwrap_or(false)
            {
                strides_b[i] = b_stride[b_i];
            }
        }
    }
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
