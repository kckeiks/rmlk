use crate::error::Result;
use crate::kernels::activation;
use cudarc::cudnn::{sys, CudnnDataType};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub fn compute<T>(
    stream: &Arc<CudaStream>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    activation::compute(
        stream,
        (alpha, beta),
        x_data,
        x_shape,
        x_stride,
        y_data,
        sys::cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
        sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        f64::MAX,
    )
}

#[cfg(test)]
mod tests {
    use super::compute;
    use approx::assert_abs_diff_eq;
    use cudarc::driver::{CudaContext, CudaSlice};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn run_sigmoid_test(input: &[f32], shape: &[i32], stride: &[i32], expected: &[f32]) {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let input_dev: CudaSlice<f32> = stream.clone_htod(input).unwrap();

        let mut output_dev: CudaSlice<f32> = unsafe { stream.alloc(input.len()).unwrap() };

        compute::<f32>(
            &stream,
            (1.0, 0.0),
            &input_dev,
            shape,
            stride,
            &mut output_dev,
        )
        .unwrap();

        let output = stream.clone_dtoh(&output_dev).unwrap();

        for (o, e) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*o, *e, epsilon = 1e-6);
        }
    }

    #[test]
    fn sigmoid_typical_values() {
        let input = [0.0, 1.0, -1.0, 3.0];
        let expected: Vec<f32> = input.iter().map(|&x| sigmoid(x)).collect();
        let shape = &[1, 4, 1, 1];
        let stride = &[4, 1, 1, 1];
        run_sigmoid_test(&input, shape, stride, &expected);
    }

    #[test]
    fn sigmoid_extreme_values() {
        let input = [10.0, -10.0, 100.0, -100.0];
        let expected: Vec<f32> = input.iter().map(|&x| sigmoid(x)).collect();
        let shape = &[1, 4, 1, 1];
        let stride = &[4, 1, 1, 1];
        run_sigmoid_test(&input, shape, stride, &expected);
    }

    #[test]
    fn sigmoid_batched_matrix() {
        let input = [0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 1e-6, -1e-6];
        let expected: Vec<f32> = input.iter().map(|&x| sigmoid(x)).collect();
        let shape = &[2, 2, 2, 1];
        let stride = &[4, 2, 1, 1];
        run_sigmoid_test(&input, shape, stride, &expected);
    }
}
