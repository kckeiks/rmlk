use crate::ptx::MUL;

pub const MODULE_NAME: &str = "mul";
pub const FWD_FN_NAMES: [&'static str; 3] = ["mul_fwd_f16", "mul_fwd_f32", "mul_fwd_f64"];
pub const PTX_SRC: &str = MUL;

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
        let x_dev_ptr = stream.memcpy_stod(&vec![3.0, 20.0, 4.0, 1.5]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_dev_ptr = stream.memcpy_stod(&vec![2.0, 2.0, 2.0, 4.0]).unwrap();

        let f = utils::load_kernel(&ctx.clone(), Op::Mul, DataType::Float).unwrap();

        let output_shape = vec![2, 2];
        let output_len = output_shape.iter().copied().product();
        let mut out_data = stream.alloc_zeros(output_len).unwrap();

        let mut info = create_info_buffer(&output_shape, &x_stride, &y_stride);

        unsafe {
            compute::<f32>(
                stream.clone(),
                f,
                output_shape.len(),
                &mut info,
                &x_dev_ptr,
                &y_dev_ptr,
                &mut out_data,
            )
            .unwrap();

            let result = stream.memcpy_dtov(&out_data).unwrap();

            assert_eq!(result, vec![6.0, 40.0, 8.0, 6.0])
        }
    }
}
