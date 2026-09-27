use crate::error::Result;
use crate::ptx::COS;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = COS;

#[derive(Debug)]
pub enum CosKernel {
    CosFwdF16,
    CosFwdF32,
    CosFwdF64,
}

impl From<CosKernel> for &'static str {
    fn from(value: CosKernel) -> Self {
        match value {
            CosKernel::CosFwdF16 => "cos_fwd_f16",
            CosKernel::CosFwdF32 => "cos_fwd_f32",
            CosKernel::CosFwdF64 => "cos_fwd_f64",
        }
    }
}

pub fn load_kernel(ctx: Arc<CudaContext>, kernel_name: CosKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::cos::{load_kernel, CosKernel};
    use crate::kernels::unary::compute;
    use crate::utils;
    use approx::assert_relative_eq;
    use cudarc::driver::CudaContext;
    use std::f32::consts;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);

        let x_data = stream
            .clone_htod(&vec![
                0.0,
                consts::FRAC_PI_2,
                consts::PI,
                consts::FRAC_PI_3,
                consts::FRAC_PI_4,
                consts::FRAC_PI_6,
            ])
            .unwrap();

        let f = load_kernel(ctx.clone(), CosKernel::CosFwdF32).unwrap();

        let output_shape = [2, 3];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().copied().product())
            .unwrap();

        unsafe {
            compute::<f32>(stream.clone(), f, &x_data, &mut out_data).unwrap();
        }
        let result = stream.clone_dtoh(&out_data).unwrap();
        let expected = [1.0, 0.0, -1.0, 0.5, consts::FRAC_1_SQRT_2, 0.8660254];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-6);
        }
    }
}
