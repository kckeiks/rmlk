use num_traits::{Num, ToPrimitive};
use std::cmp;
use std::ops::AddAssign;
use crate::core::error;
use crate::core::error::InternalError;

pub fn compute_stride<T: Num + Copy + AddAssign>(shape: &[T], stride: &mut [T]) {
    let ndims = shape.len();

    debug_assert_eq!(ndims, stride.len());

    stride[ndims - 1] = T::one();
    for i in (0..ndims - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

// This function assumes that a and b are compatible for broadcasting.
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
                .unwrap_or(true)
            {
                strides_a[i] = a_stride[a_i];
            }
        }

        if let Some(b_i) = i.checked_sub(mid - b_len) {
            if i.checked_sub(mid - a_len)
                .map(|idx| a_shape[idx] == b_shape[b_i] || a_shape[idx] == 1)
                .unwrap_or(true)
            {
                strides_b[i] = b_stride[b_i];
            }
        }
    }
}

pub fn compute_broadcast_stride_from_output_shape(
    a_shape: &[usize],
    a_stride: &[usize],
    output_shape: &[usize],
    broadcast_stride: &mut [usize],
) {
    assert!(a_shape.len() > 0 || output_shape.len() > 0);

    let a_ndims = a_shape.len();
    let ndims = broadcast_stride.len();

    for i in (0..ndims).rev() {
        if let Some(a_i) = i.checked_sub(ndims - a_ndims) {
            if a_shape[a_i] == output_shape[i] && a_shape[a_i] != 1 {
                broadcast_stride[i] = a_stride[a_i];
            }
        }
    }
}

// Assumes the dst buffer is the size of a or b, whichever is larger.
pub fn compute_broadcast_output_shape(a: &[usize], b: &[usize], dst: &mut [usize]) -> bool {
    let ndims = dst.len();

    debug_assert_eq!(ndims, cmp::max(a.len(), b.len()));

    // Compute offsets for aligning shorter arrays with the destination.
    let a_offset = ndims - a.len();
    let b_offset = ndims - b.len();

    for i in 0..ndims {
        // Fill missing dimensions with 1.
        let a_dim = i
            .checked_sub(a_offset)
            .map(|idx| a.get(idx).copied().unwrap_or(1))
            .unwrap_or(1);
        let b_dim = i
            .checked_sub(b_offset)
            .map(|idx| b.get(idx).copied().unwrap_or(1))
            .unwrap_or(1);

        // Check broadcasting rules.
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return false;
        }

        dst[i] = cmp::max(a_dim, b_dim);
    }

    true
}

pub struct DataIterator<'a, T> {
    shape: &'a [usize],
    stride: &'a [usize],
    data: &'a [T],
    current: usize,
    rank: usize,
}

impl<'a, T> DataIterator<'a, T> {
    pub fn new(shape: &'a [usize], stride: &'a [usize], data: &'a [T]) -> Self {
        debug_assert_eq!(shape.len(), stride.len());

        let rank = shape.iter().product::<usize>();
        Self {
            shape,
            stride,
            data,
            rank,
            current: 0,
        }
    }
}

impl<'a, T> Iterator for DataIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.data.len() {
            return None;
        }

        let old_current = self.current;
        self.current += 1;

        match self.shape.len() {
            0 => self.data.get(old_current),
            _ => {
                let mut i = 0;
                let mut tmp_i = old_current;
                for (d_i, dim) in self.shape.iter().enumerate().rev() {
                    let norm_i = tmp_i % dim;
                    i += norm_i * self.stride[d_i];
                    tmp_i /= dim;
                }

                self.data.get(i)
            }
        }
    }
}

pub fn normalize_index(index: i64, size: usize) -> error::Result<usize> {
    let norm_index = match index < 0 {
        true => size
            .checked_sub(
                index
                    .unsigned_abs()
                    .to_usize()
                    .expect("the runtime to be running in a `64-bit` system"),
            )
            .ok_or(InternalError::InvalidAxis { axis: index })?,
        false => {
            index
                .to_usize()
                .expect("the runtime to be running in a `64-bit` system")
                % size
        }
    };

    Ok(norm_index)
}