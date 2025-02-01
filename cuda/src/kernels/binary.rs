use crate::error::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use std::sync::Arc;

/// Launches a CUDA kernel that performs an element-wise operation with broadcasting support.
///
/// Supports **multidirectional (NumPy-style) broadcasting** for inputs of different shapes.
///
/// # Safety
/// - The `func` CUDA function must accept the following arguments in that order:
///   - The number of elements.
///   - The number of dimensions `ndims`.
///   - The `info_buffer` **must contain exactly `3 * ndims` elements**, structured as:
///      - First `ndims` entries: **Shape for `c`**.
///      - Next `ndims` entries: **Strides for `a`**.
///      - Last `ndims` entries: **Strides for `b`**.
///   - The lhs data slice `a`.
///   - The rhs data slice `b`.
///   - The output data slice `c`.
///
/// # Panics
/// - Panics if info buffer does not equal to 3 * `ndims`.
/// - Panics if output slice does not have the expected size based on the output shape.
pub unsafe fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    ndims: usize,
    info_buffer: &[usize],
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(3 * ndims, info_buffer.len());

    // Unfortunately, the asynchronous API only accepts owned vectors.
    let info = device.htod_copy(info_buffer.to_vec())?;

    let elem_count: usize = info_buffer[..ndims].iter().product();

    assert_eq!(elem_count, c.len());

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (elem_count, ndims, &info, a, b, c);

    unsafe { func.launch(config, params)? };

    Ok(())
}

#[cfg(test)]
pub fn create_info_buffer(c_shape: &[usize], a_stride: &[usize], b_stride: &[usize]) -> Vec<usize> {
    let ndims = c_shape.len();
    let mut info_buffer = vec![0usize; 3 * ndims];

    info_buffer[..ndims].copy_from_slice(c_shape);
    info_buffer[ndims..2 * ndims].copy_from_slice(a_stride);
    info_buffer[2 * ndims..].copy_from_slice(b_stride);

    info_buffer
}
