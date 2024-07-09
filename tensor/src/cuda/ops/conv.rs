use crate::cuda::data::CudaData;
use crate::Result;
use crate::{Context, Error};
use cudarc::cudnn;
use cudarc::cudnn::ConvForward;
use cudarc::driver::CudaDevice;
use rmlk_ir::DataType;
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = cudnn::Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    let input = ctx.get_input(0)?;
    let filter = ctx.get_input(1)?;

    if input.shape().len() > 4 || filter.shape().len() > 4 {
        return Err(Error::InvalidInputDimensions);
    }

    // Todo: validate shape.
    let shape: [i32; 4] = input
        .shape()
        .iter()
        .map(|dim| *dim as i32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    // Todo: validate shape.
    let filter_shape: [i32; 4] = filter
        .shape()
        .iter()
        .map(|dim| *dim as i32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    // Todo: Handle groups, dilation and stride.
    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d.
    let height = ((shape[2] - filter_shape[2] + 2 * 1) / 1) + 1;
    let width = ((shape[3] - filter_shape[3] + 2 * 1) / 1) + 1;

    let out_shape = [shape[0], filter_shape[1], height, width];
    let out_size = out_shape.iter().product::<i32>();
    println!("{out_shape:?}, {out_size}");
    // Todo: validate that input and filter shape are compatible. Follow the spec.

    match *input.dtype() {
        DataType::Float => {
            // Todo: handle this data and move it to device.
            let input_slice = input.data().unwrap().f32()?;
            let input_desc = cudnn
                .create_4d_tensor::<f32>(cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, shape)
                .map_err(|_| Error::CudnnInternal)?;

            let filter_slice = filter.data().unwrap().f32()?;
            let filter_desc = cudnn
                .create_4d_filter(
                    cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    filter_shape,
                )
                .map_err(|_| Error::CudnnInternal)?;

            let conv = cudnn
                .create_conv2d::<f32>(
                    // Todo: Get this from IR.
                    [1; 2],
                    [1; 2],
                    [1; 2],
                    cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                )
                .map_err(|_| Error::CudnnInternal)?;

            // Todo: Calculate the length of the output.
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
