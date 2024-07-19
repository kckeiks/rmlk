use crate::cuda::data::CudaData;
use crate::Result;
use crate::{Context, Error};
use cudarc::cudnn::{sys, ActivationForward, Cudnn};
use cudarc::driver::CudaDevice;
use rmlk_ir::DataType;
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    let x = ctx.get_input(0)?;

    let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
    let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

    match x.dtype() {
        DataType::Float => {
            let x_data = x.data().unwrap().f32().unwrap();
            let x_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let output_size = x.shape().iter().product();
            let mut y_data = device
                .alloc_zeros::<f32>(output_size)
                .map_err(|_| Error::CudnnInternal)?;
            let y_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let activation_desc = cudnn
                .create_activation::<f32>(
                    sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                    sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    f64::MAX,
                )
                .map_err(|_| Error::CudnnInternal)?;

            let op = ActivationForward {
                act: &activation_desc,
                x: &x_desc,
                y: &y_desc,
            };

            op.launch((1.0, 0.0), x_data, &mut y_data)
                .map_err(|_| Error::CudnnInternal)?;
        }
        _ => {
            todo!()
        }
    }

    Ok(())
}
