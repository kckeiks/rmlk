use crate::cuda::data::CudaData;
use crate::kernel::ConvAttributes;
use crate::Result;
use crate::{Context, Error};
use cudarc::cudnn;
use cudarc::cudnn::ConvForward;
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
        _ => todo!(),
    }

    Ok(())
}
