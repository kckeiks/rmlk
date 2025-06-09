use crate::error::Result;
use crate::ptx::TRILU;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use std::sync::Arc;
pub const MODULE_NAME: &str = "trilu";
pub const FWD_FN_NAMES: &[&'static str] = &[
    "trilu_fwd_f16",
    "trilu_fwd_f32",
    "trilu_fwd_f64",
    "trilu_fwd_i32",
];
pub const PTX_SRC: &str = TRILU;

pub unsafe fn compute<T>(
    stream: Arc<CudaStream>,
    func: CudaFunction,
    upper: bool,
    k: i64,
    rank: usize,
    info_buffer: &[usize],
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(rank * 2, info_buffer.len());
    assert_eq!(input.len(), output.len());

    let info = stream.memcpy_stod(info_buffer)?;

    let elem_count: usize = info_buffer[..rank].iter().product();

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&elem_count)
            .arg(&rank)
            .arg(&upper)
            .arg(&k)
            .arg(&info)
            .arg(input)
            .arg(output)
            .launch(config)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils;
    use cudarc::driver::CudaContext;
    use rmlk_schema::{DataTypeMap, Op};

    fn launch_trilu_test<T>(input: &[T], shape: &[usize], upper: bool, k: i64) -> Vec<T>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr + DataTypeMap + Clone + Unpin,
    {
        assert_eq!(shape.len(), 3);

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let func = utils::load_kernel(&ctx, Op::Trilu, T::data_type()).unwrap();

        let mut strides = vec![0; 3];
        utils::calculate_stride(shape, &mut strides);

        let mut info_buffer = Vec::new();
        info_buffer.extend_from_slice(shape);
        info_buffer.extend_from_slice(strides.as_ref());

        let input = stream.memcpy_stod(input).unwrap();

        let mut output = stream.alloc_zeros::<T>(input.len()).unwrap();

        unsafe {
            compute(
                stream.clone(),
                func,
                upper,
                k,
                3,
                &info_buffer,
                &input,
                &mut output,
            )
            .unwrap();
        }

        stream.memcpy_dtov(&output).unwrap()
    }

    #[test]
    fn test_upper_k0_f32() {
        let shape = [1, 3, 3];
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected = vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0];

        let out = launch_trilu_test(&input, &shape, true, 0);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_upper_k1_f32() {
        let shape = [1, 3, 3];
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected = vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0];

        let out = launch_trilu_test(&input, &shape, true, 1);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_lower_k_neg1_f32() {
        let shape = [1, 3, 3];
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected = vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0];

        let out = launch_trilu_test(&input, &shape, false, -1);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_rect_upper_f64() {
        let shape = [1, 2, 4];
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0];

        let out = launch_trilu_test(&input, &shape, true, 0);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_batch_lower_i64() {
        let shape = [2, 2, 2];
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let expected = vec![1, 0, 3, 4, 5, 0, 7, 8];

        let out = launch_trilu_test(&input, &shape, false, 0);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_full_keep_upper_f32() {
        let shape = [1, 2, 3];
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = vec![0.0; input.len()];

        let out = launch_trilu_test(&input, &shape, true, 3);
        assert_eq!(out, expected);
    }
}
