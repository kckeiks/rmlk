use crate::ptx::REDUCE_MEAN;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
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
pub unsafe fn compute<T>(
    stream: Arc<CudaStream>,
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

    let axes = stream.memcpy_stod(axes)?;
    let info = stream.memcpy_stod(tensor_info)?;

    let elem_count = output.len();

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let axes_len = axes.len();

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&axes)
            .arg(&axes_len)
            .arg(&rank)
            .arg(&info)
            .arg(input)
            .arg(&reduced_dim_prod)
            .arg(&elem_count)
            .arg(output)
            .launch(config)?
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::kernels::reduce_mean::compute;
    use crate::utils;
    use cudarc::driver::CudaContext;
    use rmlk_schema::{DataType, Op};

    pub fn create_info_buffer(shape: &[usize], stride: &[usize]) -> Vec<usize> {
        let rank = shape.len();
        let mut info_buffer = vec![0usize; 2 * rank];

        info_buffer[..rank].copy_from_slice(shape);
        info_buffer[rank..2 * rank].copy_from_slice(stride);

        info_buffer
    }

    fn run_reduce_mean_test(
        x_shape: Vec<usize>,
        x_data: Vec<f32>,
        output_shape: Vec<usize>,
        axes: &[usize],
        expected: Vec<f32>,
    ) {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let f = utils::load_kernel(&ctx, Op::ReduceMean, DataType::Float).unwrap();

        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_dev_ptr = stream.memcpy_stod(&x_data).unwrap();

        let output_len = output_shape.iter().product();
        let mut output_dev_ptr = stream.alloc_zeros(output_len).unwrap();

        let info = create_info_buffer(&x_shape, &x_stride);

        let num_elements = x_shape.iter().product();

        unsafe {
            compute(
                stream.clone(),
                f,
                num_elements,
                axes,
                x_shape.len(),
                &info,
                &x_dev_ptr,
                &mut output_dev_ptr,
            )
            .unwrap();
        }

        let result = stream.memcpy_dtov(&output_dev_ptr).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_reduce_mean_first_axis() {
        run_reduce_mean_test(
            vec![2, 3],
            vec![1., 2., 3., 4., 5., 6.],
            vec![1, 3],
            &[0],
            vec![2.5, 3.5, 4.5],
        );
    }

    #[test]
    fn test_reduce_mean_last_axis() {
        run_reduce_mean_test(
            vec![2, 3],
            vec![1., 2., 3., 4., 5., 6.],
            vec![2, 1],
            &[1],
            vec![2.0, 5.0],
        );
    }

    #[test]
    fn test_reduce_mean_all_axes() {
        run_reduce_mean_test(
            vec![2, 3],
            vec![1., 2., 3., 4., 5., 6.],
            vec![1, 1],
            &[0, 1],
            vec![3.5],
        );
    }

    #[test]
    fn test_reduce_mean_first_axis_3d() {
        run_reduce_mean_test(
            vec![4, 2, 3],
            (0..24).map(|x| x as f32).collect(),
            vec![1, 2, 3],
            &[0],
            vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        );
    }

    #[test]
    fn test_reduce_mean_mid_axis_3d() {
        run_reduce_mean_test(
            vec![4, 2, 3],
            (0..24).map(|x| x as f32).collect(),
            vec![4, 1, 3],
            &[1],
            vec![
                1.5, 2.5, 3.5, 7.5, 8.5, 9.5, 13.5, 14.5, 15.5, 19.5, 20.5, 21.5,
            ],
        );
    }

    #[test]
    fn test_reduce_mean_last_axis_3d() {
        run_reduce_mean_test(
            vec![4, 2, 3],
            (0..24).map(|x| x as f32).collect(),
            vec![4, 2, 1],
            &[2],
            vec![1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0],
        );
    }

    #[test]
    fn test_reduce_mean_first_mid_axis_3d() {
        run_reduce_mean_test(
            vec![4, 2, 3],
            (0..24).map(|x| x as f32).collect(),
            vec![1, 1, 3],
            &[0, 1],
            vec![10.5, 11.5, 12.5],
        );
    }

    #[test]
    fn test_reduce_mean_first_last_axis_3d() {
        run_reduce_mean_test(
            vec![4, 2, 3],
            (0..24).map(|x| x as f32).collect(),
            vec![1, 2, 1],
            &[0, 2],
            vec![10.0, 13.0],
        );
    }

    #[test]
    fn test_reduce_mean_mid_last_axis_3d() {
        run_reduce_mean_test(
            vec![4, 2, 3],
            (0..24).map(|x| x as f32).collect(),
            vec![4, 1, 1],
            &[1, 2],
            vec![2.5, 8.5, 14.5, 20.5],
        );
    }

    #[test]
    fn test_reduce_mean_all_axes_3d() {
        run_reduce_mean_test(
            vec![4, 2, 3],
            (0..24).map(|x| x as f32).collect(),
            vec![1, 1, 1],
            &[0, 1, 2],
            vec![11.5],
        );
    }
}
