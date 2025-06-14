use crate::ptx::COS;

pub const PTX_SRC: &str = COS;

pub enum CosKernel {
    CosFwdF16,
    CosFwdF32,
    CosFwdF64,
}

impl CosKernel {
    pub fn as_str(&self) -> &'static str {
        match self {
            CosKernel::CosFwdF16 => "cos_fwd_f16",
            CosKernel::CosFwdF32 => "cos_fwd_f32",
            CosKernel::CosFwdF64 => "cos_fwd_f64",
        }
    }
}

#[cfg(test)]
mod test {
    use crate::kernels::cos::{CosKernel, PTX_SRC};
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
                0.0, 1.5707963, 3.1415927, 1.0471976, 0.7853982, 0.5235988,
            ])
            .unwrap();

        let f = utils::load_kernel_v2(&ctx, PTX_SRC, CosKernel::CosFwdF32.as_str()).unwrap();

        let output_shape = vec![2, 3];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        unsafe {
            compute::<f32>(stream.clone(), f, &x_data, &mut out_data).unwrap();
        }
        let result = stream.memcpy_dtov(&out_data).unwrap();
        let expected = vec![1.0, 0.0, -1.0, 0.5, 0.7071068, 0.8660254];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*r, *e, epsilon = 1e-6);
        }
    }
}
