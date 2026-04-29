use crate::error::Result;
use crate::ptx::MUL;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = MUL;

#[derive(Debug)]
pub enum MulKernel {
    FwdF16,
    FwdF32,
    FwdF64,
    FwdI32,
    FwdI64,
}

impl From<MulKernel> for &'static str {
    fn from(value: MulKernel) -> &'static str {
        match value {
            MulKernel::FwdF16 => "mul_fwd_f16",
            MulKernel::FwdF32 => "mul_fwd_f32",
            MulKernel::FwdF64 => "mul_fwd_f64",
            MulKernel::FwdI32 => "mul_fwd_i32",
            MulKernel::FwdI64 => "mul_fwd_i64",
        }
    }
}

pub fn load_kernel(ctx: Arc<CudaContext>, kernel_name: MulKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::binary::{compute, create_info_buffer};
    use crate::kernels::mul::{load_kernel, MulKernel};
    use crate::utils;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_dev_ptr = stream.clone_htod(&vec![3.0, 20.0, 4.0, 1.5]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_dev_ptr = stream.clone_htod(&vec![2.0, 2.0, 2.0, 4.0]).unwrap();

        let f = load_kernel(ctx.clone(), MulKernel::FwdF32).unwrap();

        let output_shape = vec![2, 2];
        let output_len = output_shape.iter().copied().product();
        let mut out_data = stream.alloc_zeros(output_len).unwrap();

        let info = create_info_buffer(&output_shape, &x_stride, &y_stride);
        let info = stream.clone_htod(&info).unwrap();

        unsafe {
            compute::<f32>(
                stream.clone(),
                f,
                output_shape.len(),
                &info,
                &x_dev_ptr,
                &y_dev_ptr,
                &mut out_data,
            )
            .unwrap();

            let result = stream.clone_dtoh(&out_data).unwrap();

            assert_eq!(result, vec![6.0, 40.0, 8.0, 6.0])
        }
    }
}
