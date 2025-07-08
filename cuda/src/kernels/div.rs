use crate::ptx::DIV;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = DIV;

#[derive(Debug)]
pub enum DivKernel {
    DivFwdF16,
    DivFwdF32,
    DivFwdF64,
    DivFwdI32,
    DivFwdI64,
}

impl From<DivKernel> for &'static str {
    fn from(value: DivKernel) -> &'static str {
        match value {
            DivKernel::DivFwdF16 => "div_fwd_f16",
            DivKernel::DivFwdF32 => "div_fwd_f32",
            DivKernel::DivFwdF64 => "div_fwd_f64",
            DivKernel::DivFwdI32 => "div_fwd_i32",
            DivKernel::DivFwdI64 => "div_fwd_i64",
        }
    }
}

pub fn load_kernel(
    ctx: Arc<CudaContext>,
    kernel_name: DivKernel,
) -> crate::error::Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::binary::{compute, create_info_buffer};
    use crate::kernels::div::{load_kernel, DivKernel};
    use crate::utils;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_on_dev = stream.memcpy_stod(&vec![3.0, 20.0, 4.0, 0.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_on_dev = stream.memcpy_stod(&vec![2.0, 2.0, 2.0, 4.0]).unwrap();

        let f = load_kernel(ctx.clone(), DivKernel::DivFwdF32).unwrap();

        let output_shape = vec![2, 2];
        let output_len = output_shape.iter().copied().product();
        let mut out_data = stream.alloc_zeros(output_len).unwrap();

        let info = create_info_buffer(&output_shape, &x_stride, &y_stride);
        let info = stream.memcpy_stod(&info).unwrap();

        unsafe {
            compute::<f32>(
                stream.clone(),
                f,
                output_shape.len(),
                &info,
                &x_on_dev,
                &y_on_dev,
                &mut out_data,
            )
            .unwrap();

            let result = stream.memcpy_dtov(&out_data).unwrap();

            assert_eq!(result, vec![1.5, 10.0, 2.0, 0.0])
        }
    }
}
