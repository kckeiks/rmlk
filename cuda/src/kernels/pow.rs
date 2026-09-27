use crate::ptx::POW;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const PTX_SRC: &str = POW;

#[derive(Debug, Copy, Clone)]
pub enum PowKernel {
    PowFwdF16U8,
    PowFwdF16U16,
    PowFwdF16U32,
    PowFwdF16U64,
    PowFwdF16I8,
    PowFwdF16I16,
    PowFwdF16I32,
    PowFwdF16I64,
    PowFwdF16F16,
    PowFwdF16F32,
    PowFwdF16F64,

    PowFwdF32U8,
    PowFwdF32U16,
    PowFwdF32U32,
    PowFwdF32U64,
    PowFwdF32I8,
    PowFwdF32I16,
    PowFwdF32I32,
    PowFwdF32I64,
    PowFwdF32F16,
    PowFwdF32F32,
    PowFwdF32F64,

    PowFwdF64U8,
    PowFwdF64U16,
    PowFwdF64U32,
    PowFwdF64U64,
    PowFwdF64I8,
    PowFwdF64I16,
    PowFwdF64I32,
    PowFwdF64I64,
    PowFwdF64F16,
    PowFwdF64F32,
    PowFwdF64F64,

    PowFwdI32U8,
    PowFwdI32U16,
    PowFwdI32U32,
    PowFwdI32U64,
    PowFwdI32I8,
    PowFwdI32I16,
    PowFwdI32I32,
    PowFwdI32I64,
    PowFwdI32F16,
    PowFwdI32F32,
    PowFwdI32F64,

    PowFwdI64U8,
    PowFwdI64U16,
    PowFwdI64U32,
    PowFwdI64U64,
    PowFwdI64I8,
    PowFwdI64I16,
    PowFwdI64I32,
    PowFwdI64I64,
    PowFwdI64F16,
    PowFwdI64F32,
    PowFwdI64F64,
}

impl From<PowKernel> for &'static str {
    fn from(value: PowKernel) -> &'static str {
        match value {
            PowKernel::PowFwdF16U8 => "pow_fwd_f16_u8",
            PowKernel::PowFwdF16U16 => "pow_fwd_f16_u16",
            PowKernel::PowFwdF16U32 => "pow_fwd_f16_u32",
            PowKernel::PowFwdF16U64 => "pow_fwd_f16_u64",
            PowKernel::PowFwdF16I8 => "pow_fwd_f16_i8",
            PowKernel::PowFwdF16I16 => "pow_fwd_f16_i16",
            PowKernel::PowFwdF16I32 => "pow_fwd_f16_i32",
            PowKernel::PowFwdF16I64 => "pow_fwd_f16_i64",
            PowKernel::PowFwdF16F16 => "pow_fwd_f16_f16",
            PowKernel::PowFwdF16F32 => "pow_fwd_f16_f32",
            PowKernel::PowFwdF16F64 => "pow_fwd_f16_f64",

            PowKernel::PowFwdF32U8 => "pow_fwd_f32_u8",
            PowKernel::PowFwdF32U16 => "pow_fwd_f32_u16",
            PowKernel::PowFwdF32U32 => "pow_fwd_f32_u32",
            PowKernel::PowFwdF32U64 => "pow_fwd_f32_u64",
            PowKernel::PowFwdF32I8 => "pow_fwd_f32_i8",
            PowKernel::PowFwdF32I16 => "pow_fwd_f32_i16",
            PowKernel::PowFwdF32I32 => "pow_fwd_f32_i32",
            PowKernel::PowFwdF32I64 => "pow_fwd_f32_i64",
            PowKernel::PowFwdF32F16 => "pow_fwd_f32_f16",
            PowKernel::PowFwdF32F32 => "pow_fwd_f32_f32",
            PowKernel::PowFwdF32F64 => "pow_fwd_f32_f64",

            PowKernel::PowFwdF64U8 => "pow_fwd_f64_u8",
            PowKernel::PowFwdF64U16 => "pow_fwd_f64_u16",
            PowKernel::PowFwdF64U32 => "pow_fwd_f64_u32",
            PowKernel::PowFwdF64U64 => "pow_fwd_f64_u64",
            PowKernel::PowFwdF64I8 => "pow_fwd_f64_i8",
            PowKernel::PowFwdF64I16 => "pow_fwd_f64_i16",
            PowKernel::PowFwdF64I32 => "pow_fwd_f64_i32",
            PowKernel::PowFwdF64I64 => "pow_fwd_f64_i64",
            PowKernel::PowFwdF64F16 => "pow_fwd_f64_f16",
            PowKernel::PowFwdF64F32 => "pow_fwd_f64_f32",
            PowKernel::PowFwdF64F64 => "pow_fwd_f64_f64",

            PowKernel::PowFwdI32U8 => "pow_fwd_i32_u8",
            PowKernel::PowFwdI32U16 => "pow_fwd_i32_u16",
            PowKernel::PowFwdI32U32 => "pow_fwd_i32_u32",
            PowKernel::PowFwdI32U64 => "pow_fwd_i32_u64",
            PowKernel::PowFwdI32I8 => "pow_fwd_i32_i8",
            PowKernel::PowFwdI32I16 => "pow_fwd_i32_i16",
            PowKernel::PowFwdI32I32 => "pow_fwd_i32_i32",
            PowKernel::PowFwdI32I64 => "pow_fwd_i32_i64",
            PowKernel::PowFwdI32F16 => "pow_fwd_i32_f16",
            PowKernel::PowFwdI32F32 => "pow_fwd_i32_f32",
            PowKernel::PowFwdI32F64 => "pow_fwd_i32_f64",

            PowKernel::PowFwdI64U8 => "pow_fwd_i64_u8",
            PowKernel::PowFwdI64U16 => "pow_fwd_i64_u16",
            PowKernel::PowFwdI64U32 => "pow_fwd_i64_u32",
            PowKernel::PowFwdI64U64 => "pow_fwd_i64_u64",
            PowKernel::PowFwdI64I8 => "pow_fwd_i64_i8",
            PowKernel::PowFwdI64I16 => "pow_fwd_i64_i16",
            PowKernel::PowFwdI64I32 => "pow_fwd_i64_i32",
            PowKernel::PowFwdI64I64 => "pow_fwd_i64_i64",
            PowKernel::PowFwdI64F16 => "pow_fwd_i64_f16",
            PowKernel::PowFwdI64F32 => "pow_fwd_i64_f32",
            PowKernel::PowFwdI64F64 => "pow_fwd_i64_f64",
        }
    }
}

pub unsafe fn compute<X, Y>(
    stream: Arc<CudaStream>,
    func: CudaFunction,
    ndims: usize,
    info_buffer: &[usize],
    a: &CudaSlice<X>,
    b: &CudaSlice<Y>,
    c: &mut CudaSlice<X>,
) -> crate::error::Result<()>
where
    X: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    Y: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(3 * ndims, info_buffer.len());

    let mut info_ptr = stream.alloc(info_buffer.len())?;
    stream.memcpy_htod(info_buffer, &mut info_ptr)?;

    let elem_count: usize = info_buffer[..ndims].iter().product();

    assert_eq!(elem_count, c.len());

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
            .arg(&ndims)
            .arg(&info_ptr)
            .arg(a)
            .arg(b)
            .arg(c)
            .launch(config)?;
    }

    Ok(())
}

pub fn load_kernel(
    ctx: &Arc<CudaContext>,
    kernel_name: PowKernel,
) -> crate::error::Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use crate::kernels::pow::{compute, load_kernel, PowKernel};
    use crate::utils;
    use approx::assert_relative_eq;
    use cudarc::cudnn::CudnnDataType;
    use cudarc::driver::{CudaContext, DeviceRepr, ValidAsZeroBits};
    use half::f16;

    fn info_for_shape(shape: &[usize]) -> Vec<usize> {
        let mut a_stride = vec![0; shape.len()];
        let mut b_stride = vec![0; shape.len()];

        utils::calculate_stride(shape, &mut a_stride);
        utils::calculate_stride(shape, &mut b_stride);

        shape
            .iter()
            .cloned()
            .chain(a_stride.iter().cloned())
            .chain(b_stride.iter().cloned())
            .collect()
    }

    fn run_test<X, Y>(kernel: PowKernel, shape: &[usize], a_host: &[X], b_host: &[Y]) -> Vec<X>
    where
        X: Copy + PartialEq + std::fmt::Debug + CudnnDataType + ValidAsZeroBits + DeviceRepr,
        Y: Copy + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let func = load_kernel(&ctx, kernel).unwrap();
        let info = info_for_shape(shape);

        let a_dev = stream.clone_htod::<X, _>(a_host).unwrap();
        let b_dev = stream.clone_htod::<Y, _>(b_host).unwrap();
        let mut c_dev = stream.alloc_zeros(a_host.len()).unwrap();

        unsafe {
            compute(
                stream.clone(),
                func,
                shape.len(),
                &info,
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
            .unwrap();
        }

        stream.clone_dtoh(&c_dev).unwrap()
    }

    #[test]
    fn test_pow_fwd_f16_u8() {
        let shape = [3, 2];
        let a: Vec<f16> = vec![
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(1.5),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let b: Vec<u8> = vec![2, 3, 2, 1, 0, 2];
        let expected: Vec<f16> = vec![
            f16::from_f32(4.0),
            f16::from_f32(27.0),
            f16::from_f32(2.25),
            f16::from_f32(4.0),
            f16::from_f32(1.0),
            f16::from_f32(36.0),
        ];
        let result = run_test(PowKernel::PowFwdF16U8, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.to_f32(), e.to_f32(), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_pow_fwd_f16_i8() {
        let shape = [3, 2];
        let a: Vec<f16> = vec![
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(3.0),
            f16::from_f32(2.0),
        ];
        let b: Vec<i8> = vec![3, -1, 0, 2, 2, -2];
        let expected: Vec<f16> = vec![
            f16::from_f32(8.0),
            f16::from_f32(0.25),
            f16::from_f32(1.0),
            f16::from_f32(36.0),
            f16::from_f32(9.0),
            f16::from_f32(0.25),
        ];
        let result = run_test(PowKernel::PowFwdF16I8, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.to_f32(), e.to_f32(), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_pow_fwd_f16_i32() {
        let shape = [3, 2];
        let a: Vec<f16> = vec![
            f16::from_f32(3.0),
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
        ];
        let b: Vec<i32> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<f16> = vec![
            f16::from_f32(9.0),
            f16::from_f32(8.0),
            f16::from_f32(4.0),
            f16::from_f32(1.0),
            f16::from_f32(36.0),
            f16::from_f32(49.0),
        ];
        let result = run_test(PowKernel::PowFwdF16I32, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.to_f32(), e.to_f32(), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_pow_fwd_f16_i64() {
        let shape = [3, 2];
        let a: Vec<f16> = vec![
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
        ];
        let b: Vec<i64> = vec![6, 1, 0, 2, 3, 2];
        let expected: Vec<f16> = vec![
            f16::from_f32(64.0),
            f16::from_f32(3.0),
            f16::from_f32(1.0),
            f16::from_f32(25.0),
            f16::from_f32(216.0),
            f16::from_f32(49.0),
        ];
        let result = run_test(PowKernel::PowFwdF16I64, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.to_f32(), e.to_f32(), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_pow_fwd_f16_f16() {
        let shape = [3, 2];
        let a: Vec<f16> = vec![
            f16::from_f32(9.0),
            f16::from_f32(16.0),
            f16::from_f32(25.0),
            f16::from_f32(4.0),
            f16::from_f32(8.0),
            f16::from_f32(1.0),
        ];
        let b: Vec<f16> = vec![
            f16::from_f32(0.5),
            f16::from_f32(0.25),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
        ];
        let expected: Vec<f16> = vec![
            f16::from_f32(3.0),
            f16::from_f32(2.0),
            f16::from_f32(5.0),
            f16::from_f32(16.0),
            f16::from_f32(8.0),
            f16::from_f32(1.0),
        ];
        let result = run_test(PowKernel::PowFwdF16F16, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.to_f32(), e.to_f32(), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_pow_fwd_f16_f32() {
        let shape = [3, 2];
        let a: Vec<f16> = vec![
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
        ];
        let b: Vec<f32> = vec![3.0, 2.0, 0.5, 1.0, 0.0, -1.0];
        let expected: Vec<f16> = vec![
            f16::from_f32(8.0),
            f16::from_f32(9.0),
            f16::from_f32(2.0),
            f16::from_f32(5.0),
            f16::from_f32(1.0),
            f16::from_f32(0.142_857_15),
        ];
        let result = run_test(PowKernel::PowFwdF16F32, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.to_f32(), e.to_f32(), epsilon = 1e-2);
        }
    }

    #[test]
    fn test_pow_fwd_f16_f64() {
        let shape = [3, 2];
        let a: Vec<f16> = vec![
            f16::from_f32(16.0),
            f16::from_f32(81.0),
            f16::from_f32(64.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(1.0),
        ];
        let b: Vec<f64> = vec![0.25, 0.5, 0.5, 2.0, 1.0, 0.0];
        let expected: Vec<f16> = vec![
            f16::from_f32(2.0),
            f16::from_f32(9.0),
            f16::from_f32(8.0),
            f16::from_f32(16.0),
            f16::from_f32(5.0),
            f16::from_f32(1.0),
        ];
        let result = run_test(PowKernel::PowFwdF16F64, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.to_f32(), e.to_f32(), epsilon = 1e-2);
        }
    }

    /* ---------- f32 base --------------------------------------------------- */

    #[test]
    fn test_pow_fwd_f32_u8() {
        let shape = [3, 2];
        let a: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<u8> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<f32> = vec![4.0, 27.0, 4.0, 1.0, 36.0, 49.0];
        let result = run_test(PowKernel::PowFwdF32U8, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pow_fwd_f32_i8() {
        let shape = [3, 2];
        let a: Vec<f32> = vec![2.0, 4.0, 5.0, 3.0, 1.0, 2.0];
        let b: Vec<i8> = vec![3, -1, 2, 0, 3, -2];
        let expected: Vec<f32> = vec![8.0, 0.25, 25.0, 1.0, 1.0, 0.25];
        let result = run_test(PowKernel::PowFwdF32I8, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pow_fwd_f32_i32() {
        let shape = [3, 2];
        let a: Vec<f32> = vec![3.0, 2.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<i32> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<f32> = vec![9.0, 8.0, 4.0, 1.0, 36.0, 49.0];
        let result = run_test(PowKernel::PowFwdF32I32, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pow_fwd_f32_i64() {
        let shape = [3, 2];
        let a: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<i64> = vec![6, 1, 0, 2, 3, 2];
        let expected: Vec<f32> = vec![64.0, 3.0, 1.0, 25.0, 216.0, 49.0];
        let result = run_test(PowKernel::PowFwdF32I64, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pow_fwd_f32_f16() {
        let shape = [3, 2];
        let a: Vec<f32> = vec![9.0, 16.0, 25.0, 4.0, 8.0, 1.0];
        let b: Vec<f16> = vec![
            f16::from_f32(0.5),
            f16::from_f32(0.25),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
        ];
        let expected: Vec<f32> = vec![3.0, 2.0, 5.0, 16.0, 8.0, 1.0];
        let result = run_test(PowKernel::PowFwdF32F16, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pow_fwd_f32_f32() {
        let shape = [3, 2];
        let a: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<f32> = vec![3.0, 2.0, 0.5, 1.0, 0.0, -1.0];
        let expected: Vec<f32> = vec![8.0, 9.0, 2.0, 5.0, 1.0, 0.14285714];
        let result = run_test(PowKernel::PowFwdF32F32, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pow_fwd_f32_f64() {
        let shape = [3, 2];
        let a: Vec<f32> = vec![16.0f32, 81.0, 64.0, 4.0, 5.0, 1.0];
        let b: Vec<f64> = vec![0.25, 0.5, 0.5, 2.0, 1.0, 0.0];
        let expected: Vec<f32> = vec![2.0, 9.0, 8.0, 16.0, 5.0, 1.0];
        let result = run_test(PowKernel::PowFwdF32F64, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_pow_fwd_f64_u8() {
        let shape = [3, 2];
        let a: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<u8> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<f64> = vec![4.0, 27.0, 4.0, 1.0, 36.0, 49.0];
        let result = run_test(PowKernel::PowFwdF64U8, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pow_fwd_f64_i8() {
        let shape = [3, 2];
        let a: Vec<f64> = vec![2.0, 4.0, 5.0, 3.0, 1.0, 2.0];
        let b: Vec<i8> = vec![3, -1, 2, 0, 3, -2];
        let expected: Vec<f64> = vec![8.0, 0.25, 25.0, 1.0, 1.0, 0.25];
        let result = run_test(PowKernel::PowFwdF64I8, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pow_fwd_f64_i32() {
        let shape = [3, 2];
        let a: Vec<f64> = vec![3.0, 2.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<i32> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<f64> = vec![9.0, 8.0, 4.0, 1.0, 36.0, 49.0];
        let result = run_test(PowKernel::PowFwdF64I32, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pow_fwd_f64_i64() {
        let shape = [3, 2];
        let a: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<i64> = vec![6, 1, 0, 2, 3, 2];
        let expected: Vec<f64> = vec![64.0, 3.0, 1.0, 25.0, 216.0, 49.0];
        let result = run_test(PowKernel::PowFwdF64I64, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pow_fwd_f64_f16() {
        let shape = [3, 2];
        let a: Vec<f64> = vec![9.0, 16.0, 25.0, 4.0, 8.0, 1.0];
        let b: Vec<f16> = vec![
            f16::from_f32(0.5),
            f16::from_f32(0.25),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
        ];
        let expected: Vec<f64> = vec![3.0, 2.0, 5.0, 16.0, 8.0, 1.0];
        let result = run_test(PowKernel::PowFwdF64F16, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pow_fwd_f64_f32() {
        let shape = [3, 2];
        let a: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b: Vec<f32> = vec![3.0, 2.0, 0.5, 1.0, 0.0, -1.0];
        let expected: Vec<f64> = vec![8.0, 9.0, 2.0, 5.0, 1.0, 0.14285714285714285];
        let result = run_test(PowKernel::PowFwdF64F32, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pow_fwd_f64_f64() {
        let shape = [3, 2];
        let a: Vec<f64> = vec![16.0, 81.0, 64.0, 4.0, 5.0, 1.0];
        let b: Vec<f64> = vec![0.25, 0.5, 0.5, 2.0, 1.0, 0.0];
        let expected: Vec<f64> = vec![2.0, 9.0, 8.0, 16.0, 5.0, 1.0];
        let result = run_test(PowKernel::PowFwdF64F64, &shape, &a, &b);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_pow_fwd_i32_u8() {
        let shape = [3, 2];
        let a: Vec<i32> = vec![2, 3, 4, 5, 6, 7];
        let b: Vec<u8> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<i32> = vec![4, 27, 4, 1, 36, 49];
        let result = run_test(PowKernel::PowFwdI32U8, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i32_i8() {
        let shape = [3, 2];
        let a: Vec<i32> = vec![2, 4, 5, 3, 1, 2];
        let b: Vec<i8> = vec![3, -1, 2, 0, 3, -2];
        let expected: Vec<i32> = vec![8, 0, 25, 1, 1, 0];
        let result = run_test(PowKernel::PowFwdI32I8, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i32_i32() {
        let shape = [3, 2];
        let a: Vec<i32> = vec![3, 2, 4, 5, 6, 7];
        let b: Vec<i32> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<i32> = vec![9, 8, 4, 1, 36, 49];
        let result = run_test(PowKernel::PowFwdI32I32, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i32_i64() {
        let shape = [3, 2];
        let a: Vec<i32> = vec![2, 3, 4, 5, 6, 7];
        let b: Vec<i64> = vec![6, 1, 0, 2, 3, 2];
        let expected: Vec<i32> = vec![64, 3, 1, 25, 216, 49];
        let result = run_test(PowKernel::PowFwdI32I64, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i32_f16() {
        let shape = [3, 2];
        let a: Vec<i32> = vec![9, 16, 25, 4, 8, 1];
        let b: Vec<f16> = vec![
            f16::from_f32(0.5),
            f16::from_f32(0.25),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
        ];
        let expected: Vec<i32> = vec![3, 2, 5, 16, 8, 1];
        let result = run_test(PowKernel::PowFwdI32F16, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i32_f32() {
        let shape = [3, 2];
        let a: Vec<i32> = vec![2, 3, 4, 5, 6, 7];
        let b: Vec<f32> = vec![3.0, 2.0, 0.5, 1.0, 0.0, -1.0];
        let expected: Vec<i32> = vec![8, 9, 2, 5, 1, 0];
        let result = run_test(PowKernel::PowFwdI32F32, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i32_f64() {
        let shape = [3, 2];
        let a: Vec<i32> = vec![16, 81, 64, 4, 5, 1];
        let b: Vec<f64> = vec![0.25, 0.5, 0.5, 2.0, 1.0, 0.0];
        let expected: Vec<i32> = vec![2, 9, 8, 16, 5, 1];
        let result = run_test(PowKernel::PowFwdI32F64, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i64_u8() {
        let shape = [3, 2];
        let a: Vec<i64> = vec![2, 3, 4, 5, 6, 7];
        let b: Vec<u8> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<i64> = vec![4, 27, 4, 1, 36, 49];
        let result = run_test(PowKernel::PowFwdI64U8, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i64_i8() {
        let shape = [3, 2];
        let a: Vec<i64> = vec![2, 4, 5, 3, 1, 2];
        let b: Vec<i8> = vec![3, -1, 2, 0, 3, -2];
        let expected: Vec<i64> = vec![8, 0, 25, 1, 1, 0];
        let result = run_test(PowKernel::PowFwdI64I8, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i64_i32() {
        let shape = [3, 2];
        let a: Vec<i64> = vec![3, 2, 4, 5, 6, 7];
        let b: Vec<i32> = vec![2, 3, 1, 0, 2, 2];
        let expected: Vec<i64> = vec![9, 8, 4, 1, 36, 49];
        let result = run_test(PowKernel::PowFwdI64I32, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i64_i64() {
        let shape = [3, 2];
        let a: Vec<i64> = vec![2, 3, 4, 5, 6, 7];
        let b: Vec<i64> = vec![6, 1, 0, 2, 3, 2];
        let expected: Vec<i64> = vec![64, 3, 1, 25, 216, 49];
        let result = run_test(PowKernel::PowFwdI64I64, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i64_f16() {
        let shape = [3, 2];
        let a: Vec<i64> = vec![9, 16, 25, 4, 8, 1];
        let b: Vec<f16> = vec![
            f16::from_f32(0.5),
            f16::from_f32(0.25),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
        ];
        let expected: Vec<i64> = vec![3, 2, 5, 16, 8, 1];
        let result = run_test(PowKernel::PowFwdI64F16, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i64_f32() {
        let shape = [3, 2];
        let a: Vec<i64> = vec![2, 3, 4, 5, 6, 7];
        let b: Vec<f32> = vec![3.0, 2.0, 0.5, 1.0, 0.0, -1.0];
        let expected: Vec<i64> = vec![8, 9, 2, 5, 1, 0];
        let result = run_test(PowKernel::PowFwdI64F32, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_fwd_i64_f64() {
        let shape = [3, 2];
        let a: Vec<i64> = vec![16, 81, 64, 4, 5, 1];
        let b: Vec<f64> = vec![0.25, 0.5, 0.5, 2.0, 1.0, 0.0];
        let expected: Vec<i64> = vec![2, 9, 8, 16, 5, 1];
        let result = run_test(PowKernel::PowFwdI64F64, &shape, &a, &b);
        assert_eq!(result, expected);
    }

    // Todo: These types do not implement CudnnDataType.
    // #[test]
    // fn test_pow_fwd_f64_u16() {
    //     let a = [4.0f64, 2.0];
    //     let b = [2u16, 5u16];
    //     let expected = [16.0f64, 32.0];
    //     run_test(PowKernel::PowFwdF64U16, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f64_u32() {
    //     let a = [5.0f64];
    //     let b = [3u32];
    //     let expected = [125.0f64];
    //     run_test(PowKernel::PowFwdF64U32, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f64_u64() {
    //     let a = [2.0f64];
    //     let b = [10u64];
    //     let expected = [1024.0f64];
    //     run_test(PowKernel::PowFwdF64U64, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f64_i16() {
    //     let a = [9.0f64];
    //     let b = [-2i16];
    //     let expected = [1.0f64 / 81.0];
    //     run_test(PowKernel::PowFwdF64I16, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f32_u16() {
    //     let a = [4.0f32, 2.0];
    //     let b = [2u16, 5u16];
    //     let expected = [16.0f32, 32.0];
    //     run_test(PowKernel::PowFwdF32U16, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f32_u32() {
    //     let a = [5.0f32];
    //     let b = [3u32];
    //     let expected = [125.0f32];
    //     run_test(PowKernel::PowFwdF32U32, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f32_u64() {
    //     let a = [2.0f32];
    //     let b = [10u64];
    //     let expected = [1024.0f32];
    //     run_test(PowKernel::PowFwdF32U64, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f32_i16() {
    //     let a = [9.0f32];
    //     let b = [-2i16];
    //     let expected = [1.0f32 / 81.0];
    //     run_test(PowKernel::PowFwdF32I16, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f16_i16() {
    //     let a = [f16::from_f32(9.0)];
    //     let b = [-2i16];
    //     let expected = [f16::from_f32(1.0 / 81.0)];
    //     run_test(PowKernel::PowFwdF16I16, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f16_u16() {
    //     let a = [f16::from_f32(4.0), f16::from_f32(2.0)];
    //     let b = [2u16, 5u16];
    //     let expected = [f16::from_f32(16.0), f16::from_f32(32.0)];
    //     run_test(PowKernel::PowFwdF16U16, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f16_u32() {
    //     let a = [f16::from_f32(5.0)];
    //     let b = [3u32];
    //     let expected = [f16::from_f32(125.0)];
    //     run_test(PowKernel::PowFwdF16U32, &a, &b, &expected);
    // }
    //
    // #[test]
    // fn test_pow_fwd_f16_u64() {
    //     let a = [f16::from_f32(2.0)];
    //     let b = [10u64];
    //     let expected = [f16::from_f32(1024.0)];
    //     run_test(PowKernel::PowFwdF16U64, &a, &b, &expected);
    // }
}
