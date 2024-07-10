use crate::cuda::data::CudaData;
use crate::kernel::ConvAttributes;
use crate::{Context, Error};
use crate::{OpKernelAttributes, Result};
use cudarc::cudnn;
use cudarc::cudnn::ConvForward;
use cudarc::driver::CudaDevice;
use rmlk_ir::{AttributeType, DataType};
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = cudnn::Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    // Input data tensor.
    let x = ctx.get_input(0)?;
    // Weight tensor.
    let w = ctx.get_input(1)?;

    // Todo: Handle conv1 and conv3.
    if x.shape().len() > 4 || w.shape().len() > 4 {
        return Err(Error::InvalidInputDimensions);
    }

    let x_shape: [i32; 4] = x
        .shape()
        .iter()
        .map(|dim| *dim as i32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let attr = ctx
        .get_attributes()
        .map(|attr| ConvAttributes::try_from(attr))
        .transpose()?
        .unwrap_or_default();

    // Todo: Research this.
    // Onnx spec says a pad value will be included for each spatial axis but the
    // cudnn only accepts padding for the "height" and "weight".
    // We end up with some unused padding.
    let pads: [i32; 2] = match attr.pads {
        Some(pads) if pads.len() >= 2 => pads[..2]
            .as_ref()
            .try_into()
            .map_err(|_| Error::UnsupportedShape)?,
        _ => [0; 2],
    };
    let strides: [i32; 2] = match attr.strides {
        Some(strides) => {
            strides
                .as_ref()
                .try_into()
                .map_err(|_| Error::UnsupportedShape)?
        }
        _ => [1; 2],
    };
    let dilations: [i32; 2] = match attr.dilations {
        Some(dilations) => {
            dilations
                .as_ref()
                .try_into()
                .map_err(|_| Error::UnsupportedShape)?
        }
        _ => [1; 2],
    };
    let group = attr.group.unwrap_or(1);
    let w_shape: [i32; 4] = if let Some(shape) = attr.kernel_shape {
        [x_shape[0], x_shape[1] / group, shape[0], shape[1]]
    } else {
        w.shape()
            .iter()
            .map(|dim| *dim as i32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    };

    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d.
    let height =
        ((x_shape[2] + 2 * pads[0] - dilations[0] * (w_shape[2] - 1) - 1) / strides[0]) + 1;
    let width = ((x_shape[3] + 2 * pads[1] - dilations[1] * (w_shape[3] - 1) - 1) / strides[1]) + 1;

    let out_shape = [x_shape[0], w_shape[1], height, width];
    let out_size = out_shape.iter().product::<i32>();

    match *x.dtype() {
        DataType::Float => {
            // Todo: handle this data and move it to device.
            let input_slice = x.data().unwrap().f32()?;
            let input_desc = cudnn
                .create_4d_tensor::<f32>(
                    cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    x_shape,
                )
                .map_err(|_| Error::CudnnInternal)?;

            let filter_slice = w.data().unwrap().f32()?;
            let filter_desc = cudnn
                .create_4d_filter(cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, w_shape)
                .map_err(|_| Error::CudnnInternal)?;

            let mut conv = cudnn
                .create_conv2d::<f32>(
                    pads,
                    strides,
                    dilations,
                    cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                )
                .map_err(|_| Error::CudnnInternal)?;

            conv.set_group_count(group)
                .map_err(|_| Error::Unknown)?;

            let out_desc = cudnn
                .create_4d_tensor::<f32>(
                    cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    out_shape,
                )
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

                // Pick algorithm
                let algo = op.pick_algorithm().map_err(|_| Error::CudnnInternal)?;

                // Get workspace size
                let workspace_size = op
                    .get_workspace_size(algo.clone())
                    .map_err(|_| Error::CudnnInternal)?;
                let mut workspace = device
                    .alloc_zeros::<u8>(workspace_size)
                    .map_err(|_| Error::AllocationFailed)?;

                // Launch conv operation
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
        _ => todo!(),
    }

    Ok(())
}
