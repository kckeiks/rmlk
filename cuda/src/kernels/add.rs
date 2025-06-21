use crate::ptx::ADD;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = ADD;

#[derive(Debug)]
pub enum AddKernel {
    FwdF16,
    FwdF32,
    FwdF64,
    FwdI32,
    FwdI64,
    FwdAlphaBetaF16,
    FwdAlphaBetaF32,
    FwdAlphaBetaF64,
}

impl From<AddKernel> for &'static str {
    fn from(value: AddKernel) -> Self {
        match value {
            AddKernel::FwdF16 => "add_fwd_f16",
            AddKernel::FwdF32 => "add_fwd_f32",
            AddKernel::FwdF64 => "add_fwd_f64",
            AddKernel::FwdI32 => "add_fwd_i32",
            AddKernel::FwdI64 => "add_fwd_i64",
            AddKernel::FwdAlphaBetaF16 => "add_alpha_beta_inplace_fwd_f16",
            AddKernel::FwdAlphaBetaF32 => "add_alpha_beta_inplace_fwd_f32",
            AddKernel::FwdAlphaBetaF64 => "add_alpha_beta_inplace_fwd_f64",
        }
    }
}

pub fn load_kernel(
    ctx: Arc<CudaContext>,
    kernel_name: AddKernel,
) -> crate::error::Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::add::AddKernel;
    use crate::kernels::binary::{compute, create_info_buffer};
    use crate::utils;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_on_dev = stream.memcpy_stod(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_on_dev = stream.memcpy_stod(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let f = super::load_kernel(ctx.clone(), AddKernel::FwdF32).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let mut info = create_info_buffer(&output_shape, &x_stride, &y_stride);

        unsafe {
            compute::<f32>(
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

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }
}
