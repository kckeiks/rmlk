use crate::error::Result;
use crate::ptx::TRANSPOSE;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const PTX_SRC: &str = TRANSPOSE;

#[derive(Debug)]
pub enum TransposeKernel {
    FwdF16,
    FwdF32,
    FwdF64,
    FwdI32,
    FwdI64,
}

impl From<TransposeKernel> for &'static str {
    fn from(value: TransposeKernel) -> Self {
        match value {
            TransposeKernel::FwdF16 => "transpose_fwd_f16",
            TransposeKernel::FwdF32 => "transpose_fwd_f32",
            TransposeKernel::FwdF64 => "transpose_fwd_f64",
            TransposeKernel::FwdI32 => "transpose_fwd_i32",
            TransposeKernel::FwdI64 => "transpose_fwd_i64",
        }
    }
}

pub fn load_kernel(ctx: Arc<CudaContext>, kernel_name: TransposeKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

pub unsafe fn compute<T>(
    stream: Arc<CudaStream>,
    func: CudaFunction,
    rank: usize,
    info: &CudaSlice<usize>,
    perm: &CudaSlice<usize>,
    input_data: &CudaSlice<T>,
    output_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(input_data.len(), output_data.len());

    let elem_count = output_data.len();

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

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&elem_count)
            .arg(&rank)
            .arg(info)
            .arg(perm)
            .arg(input_data)
            .arg(output_data)
            .launch(config)?;
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils;
    use rmlk_schema::DataTypeMap;

    /// Helper to launch the Transpose kernel and return the result.
    fn launch_transpose_test<T>(
        input: &[T],
        input_shape: &[usize],
        perm: &[usize],
        kernel_type: TransposeKernel,
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
        let rank = input_shape.len();
        assert_eq!(perm.len(), rank, "perm.len() must match rank");

        let output_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p as usize]).collect();

        let mut input_strides = vec![0; rank];
        if rank > 0 {
            utils::calculate_stride(input_shape, &mut input_strides);
        }

        let mut output_strides = vec![0; rank];
        if rank > 0 {
            utils::calculate_stride(&output_shape, &mut output_strides);
        }

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let func = load_kernel(ctx, kernel_type).unwrap();

        let mut info_buffer = Vec::with_capacity(rank * 3);
        info_buffer.extend_from_slice(&output_shape);
        info_buffer.extend_from_slice(&input_strides);
        info_buffer.extend_from_slice(&output_strides);
        let info = stream.clone_htod(&info_buffer).unwrap();

        let d_perm = stream.clone_htod(perm).unwrap();
        let d_input = stream.clone_htod(input).unwrap();

        let elem_count = output_shape.iter().product::<usize>();

        let mut d_output = stream.alloc_zeros::<T>(elem_count).unwrap();

        unsafe {
            compute(
                stream.clone(),
                func,
                rank,
                &info,
                &d_perm,
                &d_input,
                &mut d_output,
            )
            .unwrap();
        }

        stream.clone_dtoh(&d_output).unwrap()
    }

    #[test]
    fn test_transpose_scalar() {
        let input: [f32; 1] = [99.0];
        let shape: &[usize] = &[];
        let perm: &[usize] = &[];
        let out = launch_transpose_test(&input, shape, perm, TransposeKernel::FwdF32);
        assert_eq!(out, input);
    }

    #[test]
    fn test_transpose_identity_1d() {
        let input_shape = &[5];
        let perm = &[0];
        let input = vec![10, 20, 30, 40, 50];
        let out = launch_transpose_test(&input, input_shape, perm, TransposeKernel::FwdI32);
        assert_eq!(out, input);
    }

    #[test]
    fn test_transpose_reverse_2d() {
        let input_shape = &[2, 3];
        let perm = &[1, 0];
        let input = vec![0, 1, 2, 3, 4, 5];
        let expected = vec![0, 3, 1, 4, 2, 5];
        let out = launch_transpose_test(&input, input_shape, perm, TransposeKernel::FwdF32);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_transpose_arbitrary_3d() {
        // 2×3×4 → 4×2×3 with perm=[2,0,1]
        let input_shape = &[2, 3, 4];
        let perm = &[2, 0, 1];
        let input: Vec<i32> = (0..24).collect();
        // Build expected by CPU:
        let mut expected = Vec::with_capacity(24);
        for o0 in 0..4 {
            for o1 in 0..2 {
                for o2 in 0..3 {
                    // output coords (o0,o1,o2) come from input (i0,i1,i2) = (o1, o2, o0)
                    expected.push(input[o1 * 12 + o2 * 4 + o0]);
                }
            }
        }
        let out = launch_transpose_test(&input, input_shape, perm, TransposeKernel::FwdI32);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_transpose_zero_length() {
        let input_shape = &[0, 5];
        let perm = &[0, 1];
        let input: Vec<i32> = vec![];
        let out = launch_transpose_test(&input, input_shape, perm, TransposeKernel::FwdI32);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_transpose_bad_perm_len() {
        let input_shape = &[2, 2];
        let perm = &[0]; // wrong length
        let input = vec![1, 2, 3, 4];
        // This should panic on the assert_eq!(perm.len(), rank)
        let _ = launch_transpose_test(&input, input_shape, perm, TransposeKernel::FwdI32);
    }
}
