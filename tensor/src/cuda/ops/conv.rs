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

    // Todo: validate shape.
    let shape = input
        .shape()
        .iter()
        .map(|&d| d as i32)
        .collect::<Box<[i32]>>();
    let stride = input
        .stride()
        .iter()
        .map(|&d| d as i32)
        .collect::<Box<[i32]>>();

    // Todo: validate shape.
    let filter_shape = filter
        .shape()
        .iter()
        .map(|&d| d as i32)
        .collect::<Box<[i32]>>();

    match *input.dtype() {
        DataType::Float => {
            let input_slice = device
                .htod_copy(vec![1.0f32; 32 * 3 * 64 * 64 * 64])
                .unwrap();
            let input_desc = cudnn
                .create_nd_tensor::<f32>(shape.as_ref(), stride.as_ref())
                .map_err(|_| Error::CudnnInternal)?;

            let filter_slice = device
                .htod_copy(vec![1.0f32; 32 * 3 * 64 * 64 * 64])
                .unwrap();
            let filter_desc = cudnn
                .create_nd_filter(
                    cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    filter_shape.as_ref(),
                )
                .map_err(|_| Error::CudnnInternal)?;
            // Todo: where do we get these arguments?
            let conv = cudnn
                .create_convnd::<f32>(
                    &[0; 2],
                    &[1; 2],
                    &[1; 2],
                    cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                )
                .map_err(|_| Error::CudnnInternal)?;
            // Todo: Calculate the length of the output.
            let mut out_slice = device
                .alloc_zeros::<f32>(6969)
                .map_err(|_| Error::AllocationFailed)?;
            let out_desc = cudnn
                .create_nd_tensor::<f32>(&[], &[])
                .map_err(|_| Error::CudnnInternal)?;
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
                        &input_slice,
                        &filter_slice,
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
