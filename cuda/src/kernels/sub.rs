use crate::error::Result;
use crate::ptx::SUB;
use cudarc::driver::{CudaContext, CudaFunction};
use std::sync::Arc;

pub const PTX_SRC: &str = SUB;

pub enum SubKernel {
    SubFwdF16,
    SubFwdF32,
    SubFwdF64,
    SubFwdI32,
    SubFwdI64,
}

impl From<SubKernel> for &'static str {
    fn from(value: SubKernel) -> Self {
        match value {
            SubKernel::SubFwdF16 => "sub_fwd_f16",
            SubKernel::SubFwdF32 => "sub_fwd_f32",
            SubKernel::SubFwdF64 => "sub_fwd_f64",
            SubKernel::SubFwdI32 => "sub_fwd_i32",
            SubKernel::SubFwdI64 => "sub_fwd_i64",
        }
    }
}

pub fn load_kernel(ctx: Arc<CudaContext>, kernel_name: SubKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::binary::{compute, create_info_buffer};
    use crate::kernels::sub::{load_kernel, SubKernel};
    use crate::utils;
    use approx::assert_relative_eq;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_on_dev = stream.memcpy_stod(&vec![1.0, 2.0, 89.0, 4.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_on_dev = stream.memcpy_stod(&vec![1.0, 4.0, 3.0, 4.0001]).unwrap();

        let f = load_kernel(ctx.clone(), SubKernel::SubFwdF32).unwrap();

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
        let expected = vec![0.0, -2.0, 86.0, -0.0001];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-6);
        }
    }
}
