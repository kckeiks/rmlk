use crate::error::Result;
use crate::ptx::GREATER;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = GREATER;

pub enum GreaterKernel {
    GreaterFwdF16,
    GreaterFwdF32,
    GreaterFwdF64,
    GreaterFwdI32,
    GreaterFwdI64,
}

impl From<GreaterKernel> for &'static str {
    fn from(value: GreaterKernel) -> &'static str {
        match value {
            GreaterKernel::GreaterFwdF16 => "greater_fwd_f16",
            GreaterKernel::GreaterFwdF32 => "greater_fwd_f32",
            GreaterKernel::GreaterFwdF64 => "greater_fwd_f64",
            GreaterKernel::GreaterFwdI32 => "greater_fwd_i32",
            GreaterKernel::GreaterFwdI64 => "greater_fwd_i64",
        }
    }
}

pub fn load_kernel(ctx: Arc<CudaContext>, kernel_name: GreaterKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::binary::{compute_with_diff_output, create_info_buffer};
    use crate::kernels::greater::{GreaterKernel, PTX_SRC};
    use crate::utils;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_on_dev = stream.memcpy_stod(&vec![1.0, 2.0, 3.0, 4.0001]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_on_dev = stream.memcpy_stod(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let f = utils::load_kernel_v2(&ctx, PTX_SRC, GreaterKernel::GreaterFwdF32.into()).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let mut info = create_info_buffer(&output_shape, &x_stride, &y_stride);

        unsafe {
            compute_with_diff_output::<f32, bool>(
                stream.clone(),
                f,
                output_shape.len(),
                &mut info,
                &x_on_dev,
                &y_on_dev,
                &mut out_data,
            )
            .unwrap();
        }

        let result = stream.memcpy_dtov(&out_data).unwrap();

        assert_eq!(result, vec![false, false, false, true])
    }
}
