use crate::error::Result;
use crate::ptx::BINARY_ADD;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "binary_add";
pub const FWD_FN_NAMES: [&'static str; 3] = ["badd_fwd_f16", "badd_fwd_f32", "badd_fwd_f64"];
pub const PTX_SRC: &str = BINARY_ADD;

/// Launches a CUDA kernel that performs an element-wise addition with broadcasting support.
///
/// Supports **multidirectional (NumPy-style) broadcasting** for inputs of different shapes.
///
/// # Safety
/// - The `info_buffer` **must contain exactly `3 * ndims` elements**, structured as:
///   - First `ndims` entries: **Shape for `c`**.
///   - Next `ndims` entries: **Strides for `a`**.
///   - Last `ndims` entries: **Strides for `b`**.
/// - Input tensors (`a_data`, `b_data`) **must be allocated on the CUDA device** and match their corresponding shapes and strides.
///
/// # Panics
/// - Panics if `info_buffer.len() != 3 * ndims`.
pub unsafe fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    ndims: usize,
    info_buffer: &[usize],
    a_data: &CudaSlice<T>,
    b_data: &CudaSlice<T>,
    c_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(3 * ndims, info_buffer.len());

    // Unfortunately, the asynchronous API only accepts owned vectors.
    let info = device.htod_copy(info_buffer.to_vec())?;

    let elem_count: usize = info_buffer[..ndims].iter().product();
    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (elem_count, ndims, &info, a_data, b_data, c_data);

    unsafe { func.launch(config, params)? };

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::kernels::add::compute;
    use crate::utils;
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    fn create_info_buffer(c_shape: &[usize], a_stride: &[usize], b_stride: &[usize]) -> Vec<usize> {
        let ndims = c_shape.len();
        let mut info_buffer = vec![0usize; 3 * ndims];

        info_buffer[..ndims].copy_from_slice(c_shape);
        info_buffer[ndims..2 * ndims].copy_from_slice(a_stride);
        info_buffer[2 * ndims..].copy_from_slice(b_stride);

        info_buffer
    }

    #[test]
    fn test_add_f32() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let f = utils::load_kernel(&device.clone(), Op::Add, DataType::Float).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let mut info = create_info_buffer(&output_shape, &x_stride, &y_stride);

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                output_shape.len(),
                &mut info,
                &x_data,
                &y_data,
                &mut out_data,
            )
            .unwrap();
            let result = device.dtoh_sync_copy(&out_data).unwrap();

            assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
        }
    }
}
