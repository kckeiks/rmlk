use crate::core::error;
use crate::providers::cuda::activation::ActivationKernel;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub struct ReluKernel(());

impl ActivationKernel for ReluKernel {
    fn execute<T>(
        stream: &Arc<CudaStream>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_data: &mut CudaSlice<T>,
    ) -> error::Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::relu::compute(&stream, (alpha, beta), x_data, x_shape, x_stride, y_data)
            .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl ActivationKernel for NoOpKernel {
    fn execute<T: CudnnDataType>(
        _: &Arc<CudaStream>,
        _: T,
        _: T,
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &mut CudaSlice<T>,
    ) -> error::Result<()> {
        Ok(())
    }
}
