use crate::ptx::REDUCE_MEAN;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "reduce_mean";
pub const FWD_FN_NAMES: &[&str] = &[
    "reduce_mean_fwd_f16",
    "reduce_mean_fwd_f32",
    "reduce_mean_fwd_f64",
    "reduce_mean_fwd_u32",
    "reduce_mean_fwd_u64",
];

pub const PTX_SRC: &str = REDUCE_MEAN;

/// Launches a CUDA kernel that performs the reduce mean operation.
///
/// Panics if the input and output slice are not equal in size.
pub unsafe fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    reduced_dim_prod: usize,
    axes: &[usize],
    rank: usize,
    tensor_info: &[usize],
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    debug_assert!(axes.len() > 0);
    debug_assert!(input.len() > 0);
    debug_assert!(output.len() > 0);
    debug_assert!(reduced_dim_prod > 0);

    debug_assert!(rank > 0);
    debug_assert!(tensor_info.len() > 0);
    debug_assert_eq!(tensor_info.len(), 2 * rank);
    debug_assert_eq!(tensor_info[..rank].iter().product::<usize>(), input.len());

    // Unfortunately, the asynchronous API only accepts owned vectors.
    let axes = device.htod_copy(axes.to_vec())?;
    let info = device.htod_copy(tensor_info.to_vec())?;

    let elem_count = output.len();

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let axes_len = axes.len();

    let params = (
        &axes,
        axes_len,
        rank,
        &info,
        input,
        reduced_dim_prod,
        elem_count,
        output,
    );

    unsafe { func.launch(config, params)? };

    Ok(())
}

#[cfg(test)]
pub fn create_info_buffer(shape: &[usize], stride: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut info_buffer = vec![0usize; 2 * rank];

    info_buffer[..rank].copy_from_slice(shape);
    info_buffer[rank..2 * rank].copy_from_slice(stride);

    info_buffer
}

#[cfg(test)]
mod tests {
    use crate::kernels::reduce_mean::{compute, create_info_buffer};
    use crate::utils;
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_reduce_mean_first_axis() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![1, 3];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(device.clone(), f, 2, &[0], 2, &info, &x_data, &mut out_data).unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![2.5, 3.5, 4.5])
    }

    #[test]
    fn test_reduce_mean_last_axis() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![2, 1];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(device.clone(), f, 3, &[1], 2, &info, &x_data, &mut out_data).unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![2.0, 5.0])
    }

    #[test]
    fn test_reduce_mean_all_axes() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![1, 1];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(
                device.clone(),
                f,
                6,
                &[0, 1],
                2,
                &info,
                &x_data,
                &mut out_data,
            )
            .unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![3.5])
    }

    #[test]
    fn test_reduce_mean_first_axis_3d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![4, 2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy::<f32>(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![1, 2, 3];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(device.clone(), f, 4, &[0], 3, &info, &x_data, &mut out_data).unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    }

    #[test]
    fn test_reduce_mean_mid_axis_3d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![4, 2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy::<f32>(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![4, 1, 3];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(device.clone(), f, 2, &[1], 3, &info, &x_data, &mut out_data).unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(
            result,
            vec![1.5, 2.5, 3.5, 7.5, 8.5, 9.5, 13.5, 14.5, 15.5, 19.5, 20.5, 21.5]
        )
    }

    #[test]
    fn test_reduce_mean_last_axis_3d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![4, 2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy::<f32>(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![4, 2, 1];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(device.clone(), f, 3, &[2], 3, &info, &x_data, &mut out_data).unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0])
    }

    #[test]
    fn test_reduce_mean_first_mid_axis_3d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![4, 2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy::<f32>(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![1, 1, 3];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(
                device.clone(),
                f,
                8,
                &[0, 1],
                3,
                &info,
                &x_data,
                &mut out_data,
            )
            .unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![10.5, 11.5, 12.5])
    }

    #[test]
    fn test_reduce_mean_first_last_axis_3d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![4, 2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy::<f32>(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![1, 2, 1];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(
                device.clone(),
                f,
                12,
                &[0, 2],
                3,
                &info,
                &x_data,
                &mut out_data,
            )
            .unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![10.0, 13.])
    }

    #[test]
    fn test_reduce_mean_mid_last_axis_3d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![4, 2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy::<f32>(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![4, 1, 1];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(
                device.clone(),
                f,
                6,
                &[1, 2],
                3,
                &info,
                &x_data,
                &mut out_data,
            )
            .unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![2.5, 8.5, 14.5, 20.5])
    }

    #[test]
    fn test_reduce_mean_all_axes_3d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![4, 2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy::<f32>(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            ])
            .unwrap();

        let f = utils::load_kernel(&device.clone(), Op::ReduceMean, DataType::Float).unwrap();

        let output_shape = vec![1, 1, 1];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);
        unsafe {
            compute(
                device.clone(),
                f,
                24,
                &[0, 1, 2],
                3,
                &info,
                &x_data,
                &mut out_data,
            )
            .unwrap();
        };

        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![11.5])
    }
}
