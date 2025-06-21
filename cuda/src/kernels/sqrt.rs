use crate::ptx::SQRT;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = SQRT;

#[derive(Debug)]
pub enum SqrtKernel {
    FwdF16,
    FwdF32,
    FwdF64,
}

impl From<SqrtKernel> for &'static str {
    fn from(value: SqrtKernel) -> Self {
        match value {
            SqrtKernel::FwdF16 => "sqrt_fwd_f16",
            SqrtKernel::FwdF32 => "sqrt_fwd_f32",
            SqrtKernel::FwdF64 => "sqrt_fwd_f64",
        }
    }
}

pub fn load_kernel(
    ctx: Arc<CudaContext>,
    kernel_name: SqrtKernel,
) -> crate::error::Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::sqrt::SqrtKernel;
    use crate::kernels::unary::compute;
    use crate::utils;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = stream.memcpy_stod(&vec![4.0, 9.0, 16.0, 25.0]).unwrap();

        let f = super::load_kernel(ctx, SqrtKernel::FwdF32).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        unsafe {
            compute::<f32>(stream.clone(), f, &x_data, &mut out_data).unwrap();
            let result = stream.memcpy_dtov(&out_data).unwrap();

            assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0])
        }
    }
}
