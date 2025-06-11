use cudarc::cudnn::{sys, Cudnn, CudnnDataType, SoftmaxForward};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub fn compute<T>(
    stream: &Arc<CudaStream>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
    mode: sys::cudnnSoftmaxMode_t,
    algo: sys::cudnnSoftmaxAlgorithm_t,
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = Cudnn::new(stream.clone())?;

    let x_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let y_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let softmax = cudnn.create_softmax::<T>(mode)?;

    let op = SoftmaxForward {
        softmax: &softmax,
        x: &x_desc,
        y: &y_desc,
    };

    unsafe {
        op.launch((alpha, beta), algo, x_data, y_data)?;
    }

    Ok(())
}
