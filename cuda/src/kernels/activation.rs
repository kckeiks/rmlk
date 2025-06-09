use crate::error::Result;
use cudarc::cudnn::{sys, ActivationForward, Cudnn, CudnnDataType};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub(crate) fn compute<T>(
    stream: &Arc<CudaStream>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
    mode: sys::cudnnActivationMode_t,
    propagation: sys::cudnnNanPropagation_t,
    coef: f64,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = Cudnn::new(stream.clone())?;

    let x_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let y_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let activation_desc = cudnn.create_activation::<T>(mode, propagation, coef)?;

    let op = ActivationForward {
        act: &activation_desc,
        x: &x_desc,
        y: &y_desc,
    };

    unsafe {
        op.launch((alpha, beta), x_data, y_data)?;
    }

    Ok(())
}
