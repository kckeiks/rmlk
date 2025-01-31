use crate::ptx::WHERE;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "where";
pub const FWD_FN_NAMES: [&'static str; 3] = ["where_fwd_f16", "where_fwd_f32", "where_fwd_f64"];
pub const PTX_SRC: &str = WHERE;

/// Launches a CUDA kernel that performs an element-wise conditional selection (`where` operation).
///
/// This function applies the following operation:
/// ```text
/// output[i] = if z[i] != 0 { x[i] } else { y[i] }
/// ```
/// Supports **multidirectional (NumPy-style) broadcasting** for inputs of different shapes.
///
/// # Safety
/// - The `info_buffer` **must contain exactly `4 * ndims` elements**, structured as:
///   - First `ndims` entries: **Output shape**.
///   - Next `ndims` entries: **Strides for `x`**.
///   - Next `ndims` entries: **Strides for `y`**.
///   - Last `ndims` entries: **Strides for `z`**.
/// - Input tensors (`x_data`, `y_data`, `z_data`) **must be allocated on the CUDA device** and match their corresponding shapes and strides.
///
/// # Panics
/// - Panics if info buffer does not equal to 4 * `ndims`.
/// - Panics if output slice does not have the expected size based on the output shape.
pub unsafe fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    ndims: usize,
    info_buffer: &[usize],
    x_data: &CudaSlice<T>,
    y_data: &CudaSlice<T>,
    z_data: &CudaSlice<T>,
    output_data: &mut CudaSlice<T>,
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(4 * ndims, info_buffer.len());

    // Unfortunately, the asynchronous API only accepts owned vectors.
    let info = device.htod_copy(info_buffer.to_vec())?;

    let elem_count: usize = info_buffer[..ndims].iter().product();

    assert_eq!(elem_count, output_data.len());

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        elem_count,
        ndims,
        &info,
        x_data,
        y_data,
        z_data,
        output_data,
    );

    unsafe { func.launch(config, params)? };

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::kernels::whereop::compute;
    use crate::utils;
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    fn create_info_buffer(
        output_shape: &[usize],
        x_stride: &[usize],
        y_stride: &[usize],
        z_stride: &[usize],
    ) -> Vec<usize> {
        let ndims = output_shape.len();
        let mut info_buffer = vec![0usize; 4 * ndims];

        info_buffer[..ndims].copy_from_slice(output_shape);
        info_buffer[ndims..2 * ndims].copy_from_slice(x_stride);
        info_buffer[2 * ndims..3 * ndims].copy_from_slice(y_stride);
        info_buffer[3 * ndims..].copy_from_slice(z_stride);

        info_buffer
    }

    #[test]
    fn test_f32() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device.htod_copy(vec![10.0, 20.0, 30.0, 40.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let z_shape = vec![2, 2];
        let mut z_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&z_shape, &mut z_stride);
        let z_data = device.htod_copy(vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let f = utils::load_kernel(&device.clone(), Op::Where, DataType::Float).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let mut info = create_info_buffer(&output_shape, &x_stride, &y_stride, &z_stride);

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                output_shape.len(),
                &mut info,
                &x_data,
                &y_data,
                &z_data,
                &mut out_data,
            )
            .unwrap();
            let result = device.dtoh_sync_copy(&out_data).unwrap();

            assert_eq!(result, vec![10.0, 2.0, 3.0, 40.0])
        }
    }
}
