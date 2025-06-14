use crate::ptx::NEG;

pub const PTX_SRC: &str = NEG;

pub enum NegKernel {
    NegFwdF16,
    NegFwdF32,
    NegFwdF64,
    NegFwdI32,
    NegFwdI64,
}

impl NegKernel {
    pub fn as_str(&self) -> &'static str {
        match self {
            NegKernel::NegFwdF16 => "neg_fwd_f16",
            NegKernel::NegFwdF32 => "neg_fwd_f32",
            NegKernel::NegFwdF64 => "neg_fwd_f64",
            NegKernel::NegFwdI32 => "neg_fwd_i32",
            NegKernel::NegFwdI64 => "neg_fwd_i64",
        }
    }
}

#[cfg(test)]
mod test {
    use crate::kernels::neg::{NegKernel, PTX_SRC};
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

        let x_data = stream.memcpy_stod(&vec![0.0, 1.0, -2.3]).unwrap();

        let f = utils::load_kernel_v2(&ctx, PTX_SRC, NegKernel::NegFwdF32.as_str()).unwrap();

        let output_shape = vec![3];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        unsafe {
            compute::<f32>(stream.clone(), f, &x_data, &mut out_data).unwrap();
        }
        let result = stream.memcpy_dtov(&out_data).unwrap();
        let expected = vec![0.0, -1.0, 2.3];

        assert_eq!(expected, result);
    }
}
