use crate::ptx::ADD;

pub const MODULE_NAME: &str = "add";
pub const FWD_FN_NAMES: [&'static str; 5] = [
    "add_fwd_f16",
    "add_fwd_f32",
    "add_fwd_f64",
    "add_fwd_i32",
    "add_fwd_i64",
];
pub const FWD_FN_NAMES_ALPHA_BETA_INPLACE: [&'static str; 3] = [
    "add_alpha_beta_inplace_fwd_f16",
    "add_alpha_beta_inplace_fwd_f32",
    "add_alpha_beta_inplace_fwd_f64",
];
pub const PTX_SRC: &str = ADD;

#[cfg(test)]
mod test {
    use crate::kernels::binary::{compute, create_info_buffer};
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
        let x_on_dev = stream.memcpy_stod(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_on_dev = stream.memcpy_stod(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let f = utils::load_kernel(&ctx, Op::Add, DataType::Float).unwrap();

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
