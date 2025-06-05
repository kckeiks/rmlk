use crate::providers::cuda::activation::ActivationKernel;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub struct SigmoidKernel(());

impl ActivationKernel for SigmoidKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_data: &mut CudaSlice<T>,
    ) -> crate::core::error::Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::sigmoid::compute(
            device,
            (alpha, beta),
            x_data,
            x_shape,
            x_stride,
            y_data,
        )
        .map_err(Into::into)
    }
}
