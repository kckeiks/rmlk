use crate::error::Result;
use crate::ptx::TRILU;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const PTX_SRC: &str = TRILU;

pub enum TriluKernel {
    FwdF16,
    FwdF32,
    FwdF64,
    FwdI32,
    FwdU32,
    FwdI64,
    FwdU64,
}

impl From<TriluKernel> for &'static str {
    fn from(value: TriluKernel) -> Self {
        match value {
            TriluKernel::FwdF16 => "trilu_fwd_f16",
            TriluKernel::FwdF32 => "trilu_fwd_f32",
            TriluKernel::FwdF64 => "trilu_fwd_f64",
            TriluKernel::FwdI32 => "trilu_fwd_i32",
            TriluKernel::FwdU32 => "trilu_fwd_u32",
            TriluKernel::FwdI64 => "trilu_fwd_i64",
            TriluKernel::FwdU64 => "trilu_fwd_u64",
        }
    }
}

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
    T: ValidAsZeroBits + DeviceRepr,
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

pub fn load_kernel(ctx: &Arc<CudaContext>, kernel_name: TriluKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils;
    use cudarc::driver::CudaContext;
    use rmlk_schema::DataTypeMap;

    fn launch_trilu_test<T>(
        trilu_kernel: TriluKernel,
        input: &[T],
        shape: &[usize],
        upper: bool,
        k: i64,
    ) -> Vec<T>
    where
        T: ValidAsZeroBits + DeviceRepr + DataTypeMap + Clone + Unpin,
    {
        assert_eq!(shape.len(), 3);

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let func = load_kernel(&ctx, trilu_kernel).unwrap();

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
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0];

        let out = launch_trilu_test(TriluKernel::FwdF32, &input, &shape, true, 0);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_upper_k1_f32() {
        let shape = [1, 3, 3];
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected: Vec<f32> = vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0];

        let out = launch_trilu_test(TriluKernel::FwdF32, &input, &shape, true, 1);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_lower_k_neg1_f32() {
        let shape = [1, 3, 3];
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected: Vec<f32> = vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0];

        let out = launch_trilu_test(TriluKernel::FwdF32, &input, &shape, false, -1);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_rect_upper_f64() {
        let shape = [1, 2, 4];
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0];

        let out = launch_trilu_test(TriluKernel::FwdF64, &input, &shape, true, 0);
        assert_eq!(&out, &expected);
    }

    #[test]
    fn test_batch_lower_i64() {
        let shape = [2, 2, 2];
        let input: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let expected: Vec<i64> = vec![1, 0, 3, 4, 5, 0, 7, 8];

        let out = launch_trilu_test(TriluKernel::FwdI64, &input, &shape, false, 0);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_full_keep_upper_f32() {
        let shape = [1, 2, 3];
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected: Vec<f32> = vec![0.0; input.len()];

        let out = launch_trilu_test(TriluKernel::FwdF32, &input, &shape, true, 3);
        assert_eq!(out, expected);
    }
}
