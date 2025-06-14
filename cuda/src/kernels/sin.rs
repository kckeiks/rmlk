use crate::ptx::SIN;

pub const PTX_SRC: &str = SIN;

pub enum SinKernel {
    SinFwdF16,
    SinFwdF32,
    SinFwdF64,
}

impl SinKernel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SinKernel::SinFwdF16 => "sin_fwd_f16",
            SinKernel::SinFwdF32 => "sin_fwd_f32",
            SinKernel::SinFwdF64 => "sin_fwd_f64",
        }
    }
}

#[cfg(test)]
mod test {
    use crate::kernels::sin::{SinKernel, PTX_SRC};
    use crate::kernels::unary::compute;
    use crate::utils;
    use approx::assert_relative_eq;
    use cudarc::driver::CudaContext;

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 3];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);

        let x_data = stream
            .memcpy_stod(&vec![
                0.0, 0.5235988, 1.5707963, 3.1415927, 2.3561945, -1.0471976,
            ])
            .unwrap();

        let f = utils::load_kernel_v2(&ctx, PTX_SRC, SinKernel::SinFwdF32.as_str()).unwrap();

        let output_shape = vec![2, 3];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        unsafe {
            compute::<f32>(stream.clone(), f, &x_data, &mut out_data).unwrap();
        }
        let result = stream.memcpy_dtov(&out_data).unwrap();
        let expected = vec![0.0, 0.5, 1.0, 0.0, 0.70710677, -0.8660254];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-6);
        }
    }
}
