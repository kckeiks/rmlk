use crate::ptx::DIV;

pub const MODULE_NAME: &str = "div";
pub const FWD_FN_NAMES: [&'static str; 3] = ["div_fwd_f16", "div_fwd_f32", "div_fwd_f64"];
pub const PTX_SRC: &str = DIV;

#[cfg(test)]
mod test {
    use crate::kernels::binary::{compute, create_info_buffer};
    use crate::utils;
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_f32() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device.htod_copy(vec![3.0, 20.0, 4.0, 0.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_data = device.htod_copy(vec![2.0, 2.0, 2.0, 4.0]).unwrap();

        let f = utils::load_kernel(&device.clone(), Op::Div, DataType::Float).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        let mut info = create_info_buffer(&output_shape, &x_stride, &y_stride);

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                output_shape.len(),
                &mut info,
                &x_data,
                &y_data,
                &mut out_data,
            )
                .unwrap();
            let result = device.dtoh_sync_copy(&out_data).unwrap();

            assert_eq!(result, vec![1.5, 10.0, 2.0, 0.0])
        }
    }
}
