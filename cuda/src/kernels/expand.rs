use crate::ptx::EXPAND;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const PTX_SRC: &str = EXPAND;

#[derive(Debug)]
pub enum ExpandKernel {
    FwdF16,
    FwdF32,
    FwdF64,
    FwdI32,
    FwdU32,
    FwdI64,
    FwdU64,
}

impl From<ExpandKernel> for &'static str {
    fn from(kernel: ExpandKernel) -> Self {
        match kernel {
            ExpandKernel::FwdF16 => "expand_fwd_f16",
            ExpandKernel::FwdF32 => "expand_fwd_f32",
            ExpandKernel::FwdF64 => "expand_fwd_f64",
            ExpandKernel::FwdI32 => "expand_fwd_i32",
            ExpandKernel::FwdU32 => "expand_fwd_u32",
            ExpandKernel::FwdI64 => "expand_fwd_i64",
            ExpandKernel::FwdU64 => "expand_fwd_u64",
        }
    }
}

pub unsafe fn compute<T>(
    stream: Arc<CudaStream>,
    func: CudaFunction,
    input_rank: usize,
    output_rank: usize,
    info: &CudaSlice<usize>,
    elem_count: usize,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
) -> crate::error::Result<()>
where
    T: ValidAsZeroBits + DeviceRepr,
{
    if elem_count == 0 {
        return Ok(());
    }

    assert!(info.len() >= 4);

    let num_threads = 128;
    let num_blocks = elem_count.div_ceil(num_threads);

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&elem_count)
            .arg(&input_rank)
            .arg(&output_rank)
            .arg(info)
            .arg(input)
            .arg(output)
            .launch(config)?
    };

    Ok(())
}

pub fn load_kernel(
    ctx: &Arc<CudaContext>,
    kernel_name: ExpandKernel,
) -> crate::error::Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::expand::{compute, ExpandKernel};
    use crate::utils;
    use cudarc::cudnn::CudnnDataType;
    use cudarc::driver::{CudaContext, DeviceRepr, ValidAsZeroBits};
    use rmlk_schema::DataTypeMap;

    fn launch_expand_test<T>(
        input: &[T],
        input_shape: &[usize],
        output_shape: &[usize],
        kernel_type: ExpandKernel,
    ) -> Vec<T>
    where
        T: CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + DataTypeMap
            + Clone
            + Unpin
            + PartialEq
            + Default,
    {
        let input_rank = input_shape.len();
        let output_rank = output_shape.len();

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let func = super::load_kernel(&ctx, kernel_type).unwrap();

        let mut input_strides = vec![0; input_rank];
        utils::calculate_stride(input_shape, &mut input_strides);

        let mut output_strides = vec![0; output_rank];
        utils::calculate_stride(output_shape, &mut output_strides);

        let mut info_buffer = Vec::new();
        info_buffer.extend_from_slice(input_shape);
        info_buffer.extend_from_slice(&input_strides);
        info_buffer.extend_from_slice(output_shape);
        info_buffer.extend_from_slice(&output_strides);
        let info = stream.clone_htod(&info_buffer).unwrap();

        let input_dev_ptr = stream.clone_htod(input).unwrap();

        let output_len = output_shape.iter().product();
        let mut output_dev_ptr = stream.alloc_zeros::<T>(output_len).unwrap();

        unsafe {
            compute(
                stream.clone(),
                func,
                input_rank,
                output_rank,
                &info,
                output_len,
                &input_dev_ptr,
                &mut output_dev_ptr,
            )
            .unwrap();
        }

        stream.clone_dtoh(&output_dev_ptr).unwrap()
    }

    #[test]
    fn test_expand_basic() {
        let input = vec![1.0f32, 2.0, 3.0];
        let input_shape = &[1, 3];
        let output_shape = &[4, 3];

        let result = launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdF32);
        let expected = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_expand_broadcast_all_dims() {
        let input = vec![7i32];
        let input_shape = &[1, 1, 1];
        let output_shape = &[2, 3, 4];

        let result = launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdI32);
        assert_eq!(result, vec![7; 2 * 3 * 4]);
    }

    #[test]
    fn test_expand_same_mid_axis() {
        let input = vec![1.0, 2.0, 3.0];
        let input_shape = &[1, 3, 1];
        let output_shape = &[2, 3, 4];

        let result = launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdF64);
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
        let result = launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdF32);
        assert_eq!(result, input);
    }

    #[test]
    fn test_expand_broadcast_middle_axis() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let input_shape = &[2, 1, 4];
        let output_shape = &[2, 3, 4];

        let result = launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdI32);

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
        let result: Vec<f32> =
            launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdF32);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_expand_promotion() {
        let input = vec![10, 20, 30];
        let input_shape = &[3];
        let output_shape = &[2, 3];

        let result = launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdI32);

        let expected = vec![10, 20, 30, 10, 20, 30];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_expand_scalar() {
        let input = vec![1.1];
        let input_shape = &[1];
        let output_shape = &[2, 3];

        let result = launch_expand_test(&input, input_shape, output_shape, ExpandKernel::FwdF64);

        let expected = vec![1.1, 1.1, 1.1, 1.1, 1.1, 1.1];
        assert_eq!(result, expected);
    }
}
