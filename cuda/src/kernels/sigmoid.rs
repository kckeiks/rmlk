use crate::error::Result;
use crate::kernels::activation;
use cudarc::cudnn::{sys, CudnnDataType};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub fn compute<T: CudnnDataType>(
    device: Arc<CudaDevice>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    activation::compute_wrapper(
        device,
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
