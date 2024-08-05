use crate::error::Error;
use crate::error::Result;
use cudarc::cudnn;
use cudarc::cudnn::{sys, ConvBiasActivationForward, ConvForward, CudnnDataType};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use num_traits::{FromPrimitive, Num};
use std::fmt::Debug;
use std::ops::AddAssign;
use std::sync::Arc;

pub fn calculate_output_shape<T>(
    x_shape: &[T],
    kernel_shape: &[T],
    pads: &[T],
    strides: &[T],
    dilations: &[T],
    y_shape: &mut [T],
) -> Result<()>
where
    T: Num + Copy + AddAssign + FromPrimitive + Debug,
    f64: From<T>,
{
    let two = T::one() + T::one();
    if y_shape.len() == 4 && kernel_shape.len() == y_shape.len() {
        // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d.
        let height =
            ((x_shape[2] + two * pads[0] - dilations[0] * (kernel_shape[2] - T::one()) - T::one())
                / strides[0])
                + T::one();
        let width =
            ((x_shape[3] + two * pads[1] - dilations[1] * (kernel_shape[3] - T::one()) - T::one())
                / strides[1])
                + T::one();

        y_shape[0] = x_shape[0];
        y_shape[1] = kernel_shape[0];
        y_shape[2] = height;
        y_shape[3] = width;
    } else if y_shape.len() == 5 && kernel_shape.len() == y_shape.len() {
        // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d.
        let depth =
            ((x_shape[0] + two * pads[0] - dilations[0] * (kernel_shape[2] - T::one()) - T::one())
                / strides[0])
                + T::one();
        let height =
            ((x_shape[2] + two * pads[1] - dilations[1] * (kernel_shape[3] - T::one()) - T::one())
                / strides[1])
                + T::one();
        let width =
            ((x_shape[3] + two * pads[2] - dilations[2] * (kernel_shape[4] - T::one()) - T::one())
                / strides[2])
                + T::one();

        y_shape[0] = x_shape[0];
        y_shape[1] = kernel_shape[0];
        y_shape[2] = depth;
        y_shape[3] = height;
        y_shape[4] = width;
    } else {
        return Err(Error::InvalidInputShapes);
    }

    Ok(())
}

// Todo: We need to figure out how to preprocess the bias input.
// Bias is expected to be in a certain shape,
// for example see https://forums.developer.nvidia.com/t/cudnn-how-to-construct-the-bias-tensor-for-convolution-layer/35223.
pub struct BiasInput<'a, T> {
    pub data: &'a CudaSlice<T>,
    pub shape: &'a [i32],
    pub stride: &'a [i32],
}

pub fn compute<T>(
    device: Arc<CudaDevice>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    w_data: &CudaSlice<T>,
    w_shape: &[i32],
    pads: &[i32],
    strides: &[i32],
    dilations: &[i32],
    group: i32,
    bias: Option<BiasInput<T>>,
    y_data: &mut CudaSlice<T>,
    y_shape: &[i32],
    y_stride: &[i32],
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = cudnn::Cudnn::new(device.clone())?;

    // Todo: the input may not have a shape.
    // When the shape is missing, it means the input can have any shape.
    // The shape of the input can be inferred by the caller of this function.
    // This may not be a problem if we stick to the invariant that
    // every input must have a shape and we must set the shape of the output
    // if it doesn't exist.
    if (x_shape.len() != 4 && x_shape.len() != 5) || (w_shape.len() != 4 && w_shape.len() != 5) {
        return Err(Error::InvalidTensorDimensions);
    }

    // Todo: handle this data and move it to device.
    let x_desc = cudnn.create_nd_tensor::<T>(&x_shape, &x_stride)?;

    // Todo: Fix this.
    // Does this cudnnTensorFormat_t handle 5d inputs?
    let w_desc =
        cudnn.create_nd_filter(cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &w_shape)?;

    // Check for optional bias input.
    // If it exists, for performance, we compute it in one single cudnn function call.
    // Todo: handle fused activation function operations.
    match bias {
        None => {
            let mut conv = cudnn.create_convnd::<T>(
                pads,
                strides,
                dilations,
                cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )?;

            conv.set_group_count(group)?;

            let y_desc = cudnn.create_nd_tensor::<T>(&y_shape, &y_stride)?;

            {
                let op = ConvForward {
                    conv: &conv,
                    x: &x_desc,
                    w: &w_desc,
                    y: &y_desc,
                };

                // Pick algorithm.
                let algo = op.pick_algorithm()?;

                // Get workspace size.
                let workspace_size = op.get_workspace_size(algo.clone())?;
                let mut workspace = device.alloc_zeros::<u8>(workspace_size)?;

                // Launch the operation.
                unsafe {
                    op.launch(
                        algo,
                        Some(&mut workspace),
                        (alpha, beta),
                        x_data,
                        w_data,
                        y_data,
                    )?;
                }
            }
        }
        Some(bias_tensor) => {
            let bias_slice = bias_tensor.data;
            let bias_shape = bias_tensor.shape;
            let bias_stride = bias_tensor.stride;

            let bias_desc = cudnn.create_nd_tensor::<T>(&bias_shape, &bias_stride)?;

            let mut conv = cudnn.create_convnd::<T>(
                pads,
                strides,
                dilations,
                cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )?;

            conv.set_group_count(group)?;

            let y_desc = cudnn.create_nd_tensor::<T>(&y_shape, &y_stride)?;

            let z_desc = cudnn.create_nd_tensor::<T>(&y_shape, &y_stride)?;
            // Todo: Do we have to actually allocate anything if we are not going to use it?
            let z_slice = device.alloc_zeros::<T>(y_shape.iter().map(|d| *d as usize).product())?;
            {
                let activation_desc = cudnn.create_activation::<T>(
                    cudarc::cudnn::sys::cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
                    // Note: For other activations,
                    // this needs to be a certain value https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#cudnnconvolutionbiasactivationforward.
                    cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    1.0,
                )?;

                let op = ConvBiasActivationForward {
                    conv: &conv,
                    act: &activation_desc,
                    x: &x_desc,
                    w: &w_desc,
                    z: &z_desc,
                    bias: &bias_desc,
                    y: &y_desc,
                };

                // This needs to be the algorithm.
                // See https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#cudnnconvolutionbiasactivationforward.
                let algo = sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

                // Get workspace size.
                let workspace_size = op.get_workspace_size(algo.clone())?;
                let mut workspace = device.alloc_zeros::<u8>(workspace_size)?;

                unsafe {
                    op.launch(
                        algo,
                        Some(&mut workspace),
                        (alpha, beta),
                        x_data,
                        w_data,
                        &z_slice,
                        bias_slice,
                        y_data,
                    )?;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::kernels::conv::{calculate_output_shape, compute, BiasInput};
    use crate::utils;
    use cudarc::driver::CudaDevice;

    #[test]
    fn test_conv_f32_2d() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![1, 1, 5, 5];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ])
            .unwrap();

        let w_shape = vec![1, 1, 3, 3];
        let w_data = device.htod_copy(vec![1.0f32; 9]).unwrap();

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        calculate_output_shape(&x_shape, &w_shape, &[1, 1], &[1, 1], &[1, 1], &mut y_shape)
            .unwrap();
        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        let mut y_data = device
            .alloc_zeros(y_shape.iter().map(|d| *d as usize).product())
            .unwrap();

        compute::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            &w_data,
            &w_shape,
            &[1, 1],
            &[1, 1],
            &[1, 1],
            1,
            None,
            &mut y_data,
            &y_shape,
            &y_stride,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();

        assert_eq!(
            result,
            vec![
                12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0,
                117.0, 81.0, 93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0,
            ]
        )
    }

    #[test]
    fn test_conv_f32_2d_bias() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![1, 1, 5, 5];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ])
            .unwrap();

        let w_shape = vec![1, 1, 3, 3];
        let w_data = device.htod_copy(vec![1.0f32; 9]).unwrap();

        let bias_shape = vec![1, 1, 1, 1];
        let mut bias_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&bias_shape, &mut bias_stride);
        let bias_data = device.htod_copy(vec![1.0f32; 1]).unwrap();

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        calculate_output_shape(&x_shape, &w_shape, &[1, 1], &[1, 1], &[1, 1], &mut y_shape)
            .unwrap();
        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        let mut y_data = device
            .alloc_zeros(y_shape.iter().map(|d| *d as usize).product())
            .unwrap();

        compute::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            &w_data,
            &w_shape,
            &[1, 1],
            &[1, 1],
            &[1, 1],
            1,
            Some(BiasInput {
                data: &bias_data,
                shape: &bias_shape,
                stride: &bias_stride,
            }),
            &mut y_data,
            &y_shape,
            &y_stride,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();

        assert_eq!(
            result,
            vec![
                13.0, 22.0, 28.0, 34.0, 25.0, 34.0, 55.0, 64.0, 73.0, 52.0, 64.0, 100.0, 109.0,
                118.0, 82.0, 94.0, 145.0, 154.0, 163.0, 112.0, 73.0, 112.0, 118.0, 124.0, 85.0,
            ]
        )
    }

    #[test]
    fn test_conv_f32_2d_bias_2() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![1, 3, 224, 224];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device
            .htod_copy(vec![1.0; x_shape.iter().map(|d| *d as usize).product()])
            .unwrap();

        let w_shape = vec![64, 3, 7, 7];
        let w_data = device
            .htod_copy(vec![1.0; w_shape.iter().map(|d| *d as usize).product()])
            .unwrap();

        let bias_shape = vec![1, 64, 1, 1];
        let mut bias_stride = vec![0; bias_shape.len()];
        utils::calculate_stride(&bias_shape, &mut bias_stride);
        let bias_data = device.htod_copy(vec![1.0f32; 64]).unwrap();

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        calculate_output_shape(&x_shape, &w_shape, &[3, 3], &[2, 2], &[1, 1], &mut y_shape)
            .unwrap();
        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        let mut y_data = device
            .alloc_zeros(y_shape.iter().map(|d| *d as usize).product())
            .unwrap();

        compute::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            &w_data,
            &w_shape,
            &[3, 3],
            &[2, 2],
            &[1, 1],
            1,
            Some(BiasInput {
                data: &bias_data,
                shape: &bias_shape,
                stride: &bias_stride,
            }),
            &mut y_data,
            &y_shape,
            &y_stride,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();
        todo!()
        // assert_eq!(
        //     result,
        //     vec![
        //         13.0, 22.0, 28.0, 34.0, 25.0, 34.0, 55.0, 64.0, 73.0, 52.0, 64.0, 100.0, 109.0,
        //         118.0, 82.0, 94.0, 145.0, 154.0, 163.0, 112.0, 73.0, 112.0, 118.0, 124.0, 85.0,
        //     ]
        // )
    }
}
