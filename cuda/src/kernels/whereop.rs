use crate::ptx::WHERE;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "where";
pub const FWD_FN_NAMES: [&'static str; 3] = ["where_fwd_f16", "where_fwd_f32", "where_fwd_f64"];
pub const PTX_SRC: &str = WHERE;

/// Executes the WHERE kernel.
///
/// Panics if:
/// - the number of dimensions in the shape and stride parameters are not the same.
/// - the info buffer's length is not equal to 4 * number of dimensions.
pub fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    x_data: &CudaSlice<T>,
    x_stride: &[usize],
    y_data: &CudaSlice<T>,
    y_stride: &[usize],
    z_data: &CudaSlice<T>,
    z_stride: &[usize],
    output_shape: &[usize],
    output_data: &mut CudaSlice<T>,
    info_buffer: &mut [usize],
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let ndims = output_shape.len();

    assert!(ndims == x_stride.len() && ndims == y_stride.len() && ndims == z_stride.len());
    assert_eq!(info_buffer.len(), 4 * ndims);

    info_buffer[..ndims].copy_from_slice(output_shape);
    info_buffer[ndims..2 * ndims].copy_from_slice(x_stride);
    info_buffer[2 * ndims..3 * ndims].copy_from_slice(y_stride);
    info_buffer[3 * ndims..].copy_from_slice(z_stride);

    // Unfortunately, the asynchronous API only accepts owned vectors.
    let info = device.htod_copy(info_buffer.to_vec())?;

    let elem_count: usize = output_shape.iter().product();
    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        elem_count,
        output_shape.len(),
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

    #[test]
    fn test_where_f32() {
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

        let mut info = vec![0; 4 * output_shape.len()];

        compute::<f32>(
            device.clone(),
            f,
            &x_data,
            &x_stride,
            &y_data,
            &y_stride,
            &z_data,
            &z_stride,
            &output_shape,
            &mut out_data,
            &mut info,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![10.0, 2.0, 3.0, 40.0])
    }
}
