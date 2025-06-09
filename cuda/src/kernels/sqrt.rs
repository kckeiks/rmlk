use crate::ptx::SQRT;

pub const MODULE_NAME: &str = "sqrt";
pub const FWD_FN_NAMES: [&'static str; 3] = ["sqrt_fwd_f16", "sqrt_fwd_f32", "sqrt_fwd_f64"];
pub const PTX_SRC: &str = SQRT;

#[cfg(test)]
mod test {
    use crate::kernels::unary::compute;
    use crate::utils;
    use cudarc::driver::CudaContext;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = stream.memcpy_stod(&vec![4.0, 9.0, 16.0, 25.0]).unwrap();

        let f = utils::load_kernel(&ctx, Op::Sqrt, DataType::Float).unwrap();

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
