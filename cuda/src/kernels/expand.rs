use crate::ptx::EXPAND;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "expand";
pub const FWD_FN_NAMES: &[&'static str] = &[
    "expand_fwd_f16",
    "expand_fwd_f32",
    "expand_fwd_f64",
    "expand_fwd_i32",
];
pub const PTX_SRC: &str = EXPAND;

pub unsafe fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    rank: usize,
    info_buffer: &[usize],
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(rank * 4, info_buffer.len());

    // Unfortunately, the asynchronous API only accepts owned vectors.
    let info = device.htod_copy(info_buffer.to_vec())?;

    let elem_count: usize = info_buffer[2 * rank..3 * rank].iter().product();

    if elem_count == 0 {
        return Ok(());
    }

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (elem_count, rank, &info, input, output);

    unsafe { func.launch(config, params)? };

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::kernels::expand::compute;
    use crate::utils;
    use cudarc::cudnn::CudnnDataType;
    use cudarc::driver::{CudaDevice, DeviceRepr, ValidAsZeroBits};
    use rmlk_schema::{DataTypeMap, Op};

    fn launch_expand_test<T>(input: &[T], input_shape: &[usize], output_shape: &[usize]) -> Vec<T>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr + DataTypeMap + Clone + Unpin + PartialEq,
    {
        assert_eq!(input_shape.len(), output_shape.len());
        let rank = input_shape.len();

        let device = CudaDevice::new(0).unwrap();
        let func = utils::load_kernel(&device, Op::Expand, T::data_type()).unwrap();

        let mut input_strides = vec![0; rank];
        utils::calculate_stride(input_shape, &mut input_strides);

        let mut output_strides = vec![0; rank];
        utils::calculate_stride(output_shape, &mut output_strides);

        let mut info_buffer = Vec::new();
        info_buffer.extend_from_slice(input_shape);
        info_buffer.extend_from_slice(&input_strides);
        info_buffer.extend_from_slice(output_shape);
        info_buffer.extend_from_slice(&output_strides);

        let input = device.htod_copy(input.to_vec()).unwrap();
        let mut output = device
            .alloc_zeros::<T>(output_shape.iter().product())
            .unwrap();

        unsafe {
            compute(
                device.clone(),
                func,
                rank,
                &info_buffer,
                &input,
                &mut output,
            )
            .unwrap();
        }

        device.dtoh_sync_copy(&output).unwrap()
    }

    #[test]
    fn test_expand_basic() {
        let input = vec![1.0f32, 2.0, 3.0];
        let input_shape = &[1, 3];
        let output_shape = &[4, 3];

        let result = launch_expand_test(&input, input_shape, output_shape);
        let expected = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_expand_scalar() {
        let input = vec![7i32];
        let input_shape = &[1, 1, 1];
        let output_shape = &[2, 3, 4];

        let result = launch_expand_test(&input, input_shape, output_shape);
        assert_eq!(result, vec![7; 2 * 3 * 4]);
    }

    #[test]
    fn test_expand_same_mid_axis() {
        let input = vec![1.0, 2.0, 3.0];
        let input_shape = &[1, 3, 1];
        let output_shape = &[2, 3, 4];

        let result = launch_expand_test(&input, input_shape, output_shape);
        let mut expected = Vec::new();
        for _ in 0..2 {
            for &v in &[1.0, 2.0, 3.0] {
                for _ in 0..4 {
                    expected.push(v);
                }
            }
        }
        assert_eq!(result, expected);
    }

    #[test]
    fn test_expand_no_broadcast() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_shape = &[2, 2];
        let output_shape = &[2, 2];
        let result = launch_expand_test(&input, input_shape, output_shape);
        assert_eq!(result, input);
    }

    #[test]
    fn test_expand_broadcast_middle_axis() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let input_shape = &[2, 1, 4];
        let output_shape = &[2, 3, 4];

        let result = launch_expand_test(&input, input_shape, output_shape);

        let expected = vec![
            1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_expand_zero_size() {
        let input = vec![];
        let input_shape = &[1, 0, 1];
        let output_shape = &[2, 0, 3];
        let result: Vec<f32> = launch_expand_test(&input, input_shape, output_shape);
        assert_eq!(result.len(), 0);
    }
}
