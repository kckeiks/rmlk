use crate::error::Result;
use crate::ptx::NEG;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = NEG;

#[derive(Debug)]
pub enum NegKernel {
    NegFwdF16,
    NegFwdF32,
    NegFwdF64,
    NegFwdI32,
    NegFwdI64,
}

impl From<NegKernel> for &'static str {
    fn from(value: NegKernel) -> Self {
        match value {
            NegKernel::NegFwdF16 => "neg_fwd_f16",
            NegKernel::NegFwdF32 => "neg_fwd_f32",
            NegKernel::NegFwdF64 => "neg_fwd_f64",
            NegKernel::NegFwdI32 => "neg_fwd_i32",
            NegKernel::NegFwdI64 => "neg_fwd_i64",
        }
    }
}

pub fn load_kernel(ctx: Arc<CudaContext>, kernel_name: NegKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::neg::{load_kernel, NegKernel};
    use crate::kernels::unary::compute;
    use crate::utils;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);

        let x_data = stream.clone_htod(&vec![0.0, 1.0, -2.3]).unwrap();

        let f = load_kernel(ctx.clone(), NegKernel::NegFwdF32).unwrap();

        let output_shape = [3];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().copied().product())
            .unwrap();

        unsafe {
            compute::<f32>(stream.clone(), f, &x_data, &mut out_data).unwrap();
        }
        let result = stream.clone_dtoh(&out_data).unwrap();
        let expected = vec![0.0, -1.0, 2.3];

        assert_eq!(expected, result);
    }
}
