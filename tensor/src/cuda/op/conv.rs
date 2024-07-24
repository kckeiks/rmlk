use crate::attribute::conv::ConvAttributes;
use crate::cuda::data::CudaData;
use crate::Result;
use crate::{Context, Error};
use cudarc::cudnn;
use cudarc::cudnn::{ConvBiasActivationForward, ConvForward};
use cudarc::driver::CudaDevice;
use log::debug;
use rmlk_ir::DataType;
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = cudnn::Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    // Input data tensor.
    let x = ctx.get_input(0)?;
    // Weight tensor.
    let w = ctx.get_input(1)?;

    // Todo: the input may not have a shape.
    // When the shape is missing, it means the input can have any shape.
    // The shape of the input can be inferred by the caller of this function.
    // This may not be a problem if we stick to the invariant that
    // every input must have a shape and we must set the shape of the output
    // if it doesn't exist.
    if (x.shape().len() != 4 && x.shape().len() != 5)
        || (w.shape().len() != 4 && w.shape().len() != 5)
    {
        return Err(Error::InvalidTensorDimensions);
    }

    // Todo: Update this to box sliced.
    let x_shape: [i32; 4] = x
        .shape()
        .iter()
        .map(|dim| *dim as i32)
        .collect::<Vec<_>>()
        .try_into()
        // Unwrap is safe because we validated the dimensions.
        .unwrap();
    let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

    let filter_dims = match x.shape().len() {
        4 => 2,
        5 => 3,
        _ => unreachable!("we already checked the dimensions of x for the supported dimensions"),
    };

    let attrs = ConvAttributes::new(
        ctx.get_attributes().ok_or(Error::MissingNodeInGraph)?,
        filter_dims,
    )?;

    let w_shape = match attrs.kernel_shape(&x_shape) {
        None => {
            w.shape()
                .iter()
                .map(|dim| *dim as i32)
                // Todo: use a scratch buffer to avoid an allocation.
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        }
        Some(shape) => shape,
    };
    let out_shape = attrs.calculate_output_shape(&x_shape, &w_shape);
    let out_size = out_shape.iter().product::<i32>();

    // Todo: Do we validate that our calculated output shape matches
    // the output's shape, if any exists?
    // For now, we validate and ignore that shape may not be present.
    let out = ctx.get_output(0)?;
    if out.shape().len() != out_shape.len() {
        return Err(Error::CudnnInternal);
    }

    // Todo: if there is no shape, simply set it.
    // Although, we probably need some other type of way
    // to flag tensors when they could have any shape.
    if out
        .shape()
        .iter()
        .zip(out_shape.iter())
        .map(|(a, b)| (*a, *b as usize))
        .find(|(a, b)| a != b)
        .is_some()
    {
        return Err(Error::OutputShapeMismatch);
    }

    let out_stride = out
        .stride()
        .iter()
        .map(|v| *v as i32)
        .collect::<Box<[i32]>>();

    debug!(
        "x_shape={x_shape:?}, \
        w_shape={w_shape:?}, \
        filter_dims={filter_dims:?}, \
        out_shape={out_shape:?}, \
        group={:?}, \
        pads={:?}, \
        strides={:?}, \
        dilations={:?}",
        attrs.group(),
        attrs.pads(),
        attrs.strides(),
        attrs.dilations()
    );

    match *x.dtype() {
        DataType::Float => {
            // Todo: handle this data and move it to device.
            let input_slice = x.data().unwrap().f32()?;
            let input_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let filter_slice = w.data().unwrap().f32()?;
            let filter_desc = cudnn
                .create_nd_filter(cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &w_shape)
                .map_err(|_| Error::CudnnInternal)?;

            // Check for optional bias input.
            // If it exists, for performance, we compute it in one single cudnn function call.
            // Todo: handle fused activation function operations.
            match ctx.get_input(2).ok() {
                None => {
                    // Todo: handle this data and move it to device.
                    let input_slice = x.data().unwrap().f32()?;
                    let input_desc = cudnn
                        .create_nd_tensor::<f32>(&x_shape, &x_stride)
                        .map_err(|_| Error::CudnnInternal)?;

                    let filter_slice = w.data().unwrap().f32()?;
                    let filter_desc = cudnn
                        .create_nd_filter(
                            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                            &w_shape,
                        )
                        .map_err(|_| Error::CudnnInternal)?;

                    let mut conv = cudnn
                        .create_convnd::<f32>(
                            attrs.pads(),
                            attrs.strides(),
                            attrs.dilations(),
                            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                        )
                        .map_err(|_| Error::CudnnInternal)?;

                    conv.set_group_count(attrs.group())
                        .map_err(|_| Error::Unknown)?;

                    let out_desc = cudnn
                        .create_nd_tensor::<f32>(&out_shape, &out_stride)
                        .map_err(|_| Error::CudnnInternal)?;
                    let mut out_slice = device
                        .alloc_zeros::<f32>(out_size as usize)
                        .map_err(|_| Error::AllocationFailed)?;
                    {
                        let op = ConvForward {
                            conv: &conv,
                            x: &input_desc,
                            w: &filter_desc,
                            y: &out_desc,
                        };

                        // Pick algorithm.
                        let algo = op.pick_algorithm().map_err(|_| Error::CudnnInternal)?;

                        // Get workspace size.
                        let workspace_size = op
                            .get_workspace_size(algo.clone())
                            .map_err(|_| Error::CudnnInternal)?;
                        let mut workspace = device
                            .alloc_zeros::<u8>(workspace_size)
                            .map_err(|_| Error::AllocationFailed)?;

                        // Launch the operation.
                        unsafe {
                            op.launch(
                                algo,
                                Some(&mut workspace),
                                (1.0, 0.0),
                                input_slice,
                                filter_slice,
                                &mut out_slice,
                            )
                            .map_err(|_| Error::CudnnInternal)?;
                        }
                    }
                    let out = ctx.get_output_mut(0)?;
                    out.init(CudaData::F32(out_slice));
                }
                Some(bias_tensor) => {
                    // Todo: We need to figure out how to preprocess the bias input.
                    // Bias is expected to be in a certain shape,
                    // for example see https://forums.developer.nvidia.com/t/cudnn-how-to-construct-the-bias-tensor-for-convolution-layer/35223.
                    let bias_slice = bias_tensor.data().unwrap().f32()?;
                    let bias_shape = bias_tensor
                        .shape()
                        .iter()
                        .map(|d| *d as i32)
                        .collect::<Box<[i32]>>();
                    let bias_stride = bias_tensor
                        .stride()
                        .iter()
                        .map(|d| *d as i32)
                        .collect::<Box<[i32]>>();
                    let bias_desc = cudnn
                        .create_nd_tensor::<f32>(&bias_shape, &bias_stride)
                        .map_err(|_| Error::CudnnInternal)?;

                    debug!("bias_shape={:?},bias_stride{:?}", bias_shape, bias_stride);

                    let mut conv = cudnn
                        .create_convnd::<f32>(
                            attrs.pads(),
                            attrs.strides(),
                            attrs.dilations(),
                            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                        )
                        .map_err(|_| Error::CudnnInternal)?;

                    conv.set_group_count(attrs.group())
                        .map_err(|_| Error::Unknown)?;

                    let out_desc = cudnn
                        .create_nd_tensor::<f32>(&out_shape, &out_stride)
                        .map_err(|_| Error::CudnnInternal)?;
                    let mut out_slice = device
                        .alloc_zeros::<f32>(out_size as usize)
                        .map_err(|_| Error::AllocationFailed)?;

                    let z_desc = cudnn
                        .create_nd_tensor::<f32>(&out_shape, &out_stride)
                        .map_err(|_| Error::CudnnInternal)?;
                    // Todo: Do we have to actually allocate anything if we are not going to use it?
                    let z_slice = device
                        .alloc_zeros::<f32>(out_size as usize)
                        .map_err(|_| Error::AllocationFailed)?;
                    {
                        let activation_desc = cudnn.create_activation::<f32>(
                            cudarc::cudnn::sys::cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
                            // Note: For other activations,
                            // this needs to be a certain value https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#cudnnconvolutionbiasactivationforward.
                            cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                            1.0
                        ).map_err(|_| Error::CudnnInternal)?;

                        let op = ConvBiasActivationForward {
                            conv: &conv,
                            act: &activation_desc,
                            x: &input_desc,
                            w: &filter_desc,
                            z: &z_desc,
                            bias: &bias_desc,
                            y: &out_desc,
                        };

                        // Pick algorithm.
                        // let algo = op.pick_algorithm().map_err(|_| Error::CudnnInternal)?;
                        let algo = cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

                        // Get workspace size.
                        let workspace_size = op
                            .get_workspace_size(algo.clone())
                            .map_err(|_| Error::CudnnInternal)?;
                        let mut workspace = device
                            .alloc_zeros::<u8>(workspace_size)
                            .map_err(|_| Error::AllocationFailed)?;

                        // Launch the operation.
                        unsafe {
                            op.launch(
                                algo,
                                Some(&mut workspace),
                                (1.0, 0.0),
                                input_slice,
                                filter_slice,
                                &z_slice,
                                bias_slice,
                                &mut out_slice,
                            )
                            .map_err(|e| {
                                debug!("cudnn error: {:?}", e.0);
                                Error::CudnnInternal
                            })?;
                        }
                    }
                    let out = ctx.get_output_mut(0)?;
                    out.init(CudaData::F32(out_slice));
                }
            }
        }
        _ => todo!(),
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::CudaKernel;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils;
    use crate::test_utils::{TestConvAttributes, TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_conv_f32_2d_bias() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 5, 5];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                    ])
                    .unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: vec![1, 1, 3, 3],
            dtype,
            data: Some(CudaData::F32(device.htod_copy(vec![1.0; 9]).unwrap())),
        };

        // Bias.
        let node_c = TestNode {
            shape: vec![1, 1, 1, 1],
            dtype,
            data: Some(CudaData::F32(device.htod_copy(vec![1.0; 1]).unwrap())),
        };

        let node_output = TestNode {
            shape,
            dtype,
            data: None,
        };

        let attributes = test_utils::create_conv_attributes(TestConvAttributes {
            dilations: Some(Box::new([1, 1])),
            group: Some(1),
            kernel_shape: None,
            pads: Some(Box::new([1, 1, 1, 1])),
            strides: Some(Box::new([1, 1])),
        });

        let params = TestParams {
            inputs: vec![node_a, node_b, node_c],
            outputs: vec![node_output],
            attributes,
            op: Op::Conv,
        };

        let (_, mut state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 3).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Conv, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(
            result,
            vec![
                13.0, 22.0, 28.0, 34.0, 25.0, 34.0, 55.0, 64.0, 73.0, 52.0, 64.0, 100.0, 109.0,
                118.0, 82.0, 94.0, 145.0, 154.0, 163.0, 112.0, 73.0, 112.0, 118.0, 124.0, 85.0,
            ]
        )
    }

    #[test]
    fn test_conv_f32_2d() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 5, 5];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                    ])
                    .unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: vec![1, 1, 3, 3],
            dtype,
            data: Some(CudaData::F32(device.htod_copy(vec![1.0; 9]).unwrap())),
        };

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let attributes = test_utils::create_conv_attributes(TestConvAttributes {
            dilations: Some(Box::new([1, 1])),
            group: Some(1),
            kernel_shape: None,
            pads: Some(Box::new([1, 1, 1, 1])),
            strides: Some(Box::new([1, 1])),
        });

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes,
            op: Op::Conv,
        };

        let (_, mut state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Conv, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(
            result,
            vec![
                12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0,
                117.0, 81.0, 93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0,
            ]
        )
    }
}
