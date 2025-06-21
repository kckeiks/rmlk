use anyhow::{anyhow, bail, Result};
use half::f16;
use num_traits::{Num, ToPrimitive};
use std::cmp;
use std::ops::AddAssign;

pub fn compute_stride<T: Num + Copy + AddAssign>(shape: &[T], stride: &mut [T]) {
    let ndims = shape.len();

    if ndims == 0 {
        return;
    }

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

    debug_assert!(ndims >= cmp::max(a.len(), b.len()));

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

#[derive(Clone)]
pub struct DataIterator<'a, T> {
    shape: &'a [usize],
    stride: &'a [usize],
    data: &'a [T],
    current: usize,
}

impl<'a, T> DataIterator<'a, T> {
    pub fn new(shape: &'a [usize], stride: &'a [usize], data: &'a [T]) -> Self {
        debug_assert_eq!(shape.len(), stride.len());

        Self {
            shape,
            stride,
            data,
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

pub fn normalize_indices(src: &[i64], dst: &mut [usize], size: usize) -> Result<()> {
    debug_assert_eq!(src.len(), dst.len());

    for (i, axis) in src.iter().copied().enumerate() {
        dst[i] = normalize_index(axis, size)?;
    }
    Ok(())
}

// This function converts `index` to a `usize` value in the range `[0, size-1]`.
pub fn normalize_index(index: i64, size: usize) -> Result<usize> {
    let norm_index = match index < 0 {
        true => size
            .checked_sub(
                index
                    .unsigned_abs()
                    .to_usize()
                    .expect("the runtime to be running in a `64-bit` system"),
            )
            .ok_or_else(|| anyhow!("axis out of bound {index}"))?,
        false => index
            .to_usize()
            .expect("the runtime to be running in a `64-bit` system"),
    };

    if norm_index > size {
        Err(anyhow!("axis out of bound {index}"))
    } else {
        Ok(norm_index)
    }
}

pub fn derive_range(start: i64, end: i64, size: usize) -> Result<(usize, usize)> {
    let norm_start = match normalize_index(start, size) {
        Ok(start) => start,
        Err(_) => {
            if start < 0 {
                0
            } else {
                bail!("axis out of bound {start}")
            }
        }
    };

    let norm_end = match normalize_index(end, size) {
        Ok(end) => end,
        Err(_) => {
            if end > 0 {
                size
            } else {
                bail!("axis out of bound {start}")
            }
        }
    };

    if norm_start > norm_end {
        bail!("invalid range start=`{start}` and end=`{end}`")
    }

    Ok((norm_start, norm_end))
}

#[inline]
pub fn write_increasing_sequence<T>(dst: &mut [T]) -> Result<()>
where
    T: From<usize>,
{
    for i in 0..dst.len() {
        dst[i] = T::try_from(i)?;
    }
    Ok(())
}

pub trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>>;
}

impl FromBytes for f16 {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for f16", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for f32 {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for f32", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for f64 {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for f64", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for i32 {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for i32", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for u32 {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for u32", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for i64 {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for i64", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for u64 {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % size_of::<Self>() != 0 {
            bail!("invalid bytes length {} for u64", bytes.len())
        }

        let mut vec = Vec::with_capacity(bytes.len() / size_of::<Self>());
        let mut chunks = bytes.chunks_exact(size_of::<Self>());
        for chunk in &mut chunks {
            let arr = chunk.try_into().unwrap();
            vec.push(Self::from_le_bytes(arr));
        }

        Ok(vec)
    }
}

impl FromBytes for bool {
    fn from_bytes(bytes: &[u8]) -> Result<Vec<Self>> {
        // Each byte is one boolean.
        let vec = bytes.iter().map(|&b| b != 0).collect();
        Ok(vec)
    }
}

pub trait FromF32 {
    fn from_f32(value: f32) -> Self;
}

impl FromF32 for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl FromF32 for f16 {
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
}

pub fn write_info<T>(shape: &[T], stride: &[T], dst: &mut [T], start: usize)
where
    T: Copy,
{
    dst[start..start + shape.len()].copy_from_slice(shape);
    dst[start + shape.len()..start + shape.len() + stride.len()].copy_from_slice(stride);
}

pub fn create_3d_shape_and_stride(shape: &[usize]) -> ([usize; 3], [usize; 3]) {
    match shape.len() {
        0 => ([1, 1, 1], [1, 1, 1]),
        1 => ([1, 1, shape[0]], [shape[0], shape[0], 1]),
        2 => ([1, shape[0], shape[1]], [shape[0] * shape[1], shape[1], 1]),
        rank => {
            let batch = shape[..rank - 2].iter().product::<usize>();
            (
                [batch, shape[rank - 2], shape[rank - 1]],
                [shape[rank - 2] * shape[rank - 1], shape[rank - 1], 1],
            )
        }
    }
}

#[cfg(test)]
mod test {
    use crate::utils::{compute_broadcast_output_shape, derive_range};
    use num_traits::ToPrimitive;

    #[test]
    fn test_derive_shape() {
        let input = vec![0; 4];

        let (start, end) = derive_range(0, input.len() as i64, input.len()).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, input.len());

        let (start, end) = derive_range(0, 2 * input.len() as i64, input.len()).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, input.len());

        let (start, end) = derive_range(
            -input.len().to_i64().unwrap(),
            2 * input.len() as i64,
            input.len(),
        )
        .unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, input.len());

        let (start, end) = derive_range(-2, input.len() as i64, input.len()).unwrap();
        assert_eq!(start, 2);
        assert_eq!(end, input.len());

        let (start, end) = derive_range(-2, -1, input.len()).unwrap();
        assert_eq!(start, 2);
        assert_eq!(end, 3);

        let (start, end) =
            derive_range(-(input.len() as i64), -(input.len() as i64), input.len()).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 0);

        assert!(derive_range(-1, -2, input.len()).is_err());
    }

    #[test]
    fn test_scalar_and_tensor() {
        let a: &[usize] = &[]; // scalar
        let b: &[usize] = &[3, 4]; // tensor
        let mut dst = [0; 2];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(result);
        assert_eq!(dst, [3, 4]);
    }

    #[test]
    fn test_tensor_and_scalar() {
        let a: &[usize] = &[3, 4];
        let b: &[usize] = &[]; // scalar
        let mut dst = [0; 2];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(result);
        assert_eq!(dst, [3, 4]);
    }

    #[test]
    fn test_equal_shapes() {
        let a = &[2, 5];
        let b = &[2, 5];
        let mut dst = [0; 2];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(result);
        assert_eq!(dst, [2, 5]);
    }

    #[test]
    fn test_broadcastable_shapes() {
        let a = &[1, 5];
        let b = &[3, 1];
        let mut dst = [0; 2];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(result);
        assert_eq!(dst, [3, 5]);
    }

    #[test]
    fn test_incompatible_shapes() {
        let a = &[2, 3];
        let b = &[3, 2];
        let mut dst = [0; 2];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(!result);
    }

    #[test]
    fn test_high_dimensional() {
        let a = &[1, 1, 10, 1];
        let b = &[4, 3, 1, 5];
        let mut dst = [0; 4];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(result);
        assert_eq!(dst, [4, 3, 10, 5]);
    }

    #[test]
    fn test_one_dim_vs_two_dim() {
        let a = &[5];
        let b = &[3, 1];
        let mut dst = [0; 2];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(result);
        assert_eq!(dst, [3, 5]);
    }

    #[test]
    fn test_scalar_and_scalar() {
        let a: &[usize] = &[];
        let b: &[usize] = &[];
        let mut dst = [];
        let expected: [usize; 0] = [];
        let result = compute_broadcast_output_shape(a, b, &mut dst);
        assert!(result);
        assert_eq!(dst, expected);
    }
}
