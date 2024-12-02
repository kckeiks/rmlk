use crate::core::device_service::DeviceService;
use crate::core::error::{InternalError, Result};
use crate::core::{ScratchAllocator, Tensor};
use crate::ops::global_average::GlobalAverageBackend;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::driver::{CudaDevice, CudaSlice};
use rmlk_schema::{DataType, Op};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct BackendHandler<T> {
    device: Arc<CudaDevice>,
    _marker: PhantomData<T>,
}

impl<T> BackendHandler<T>
where
    T: GlobalAveragePoolKernel,
{
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            _marker: PhantomData,
        }
    }
}

impl<T> GlobalAverageBackend for BackendHandler<T>
where
    T: GlobalAveragePoolKernel,
{
    type Service = Cuda;

    fn compute(
        self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        y_shape: &[usize],
        y_stride: &[usize],
        scratch_alloc: &ScratchAllocator,
    ) -> Result<<Self::Service as DeviceService>::Data> {
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;

        let y_shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(y_stride)?;

        let pads = scratch_alloc.allocate_fill(x_shape[2..].len(), 0)?;
        let strides = scratch_alloc.allocate_fill(x_shape[2..].len(), 1)?;
        let kernel_shape = &x_shape[2..];

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x.data().and_then(|data| data.f32()).ok_or_else(|| {
                InternalError::UnexpectedTensorDataType {
                    expected: DataType::Float,
                }
            })?;

            let mut y_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
                .map_err(rmlk_cuda::Error::from)?;

            T::execute::<f32>(
                self.device,
                1.0,
                0.0,
                pads,
                strides,
                &x_data,
                &x_shape,
                &x_stride,
                kernel_shape,
                &mut y_data,
                &y_shape,
                &y_stride,
            )?;

            Ok(CudaData::F32(y_data))
        } else {
            Err(InternalError::UnsupportedOpForDataType {
                op: Op::GlobalAveragePool,
                dtype: *x.dtype(),
            })
        }
    }
}

trait GlobalAveragePoolKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        pads: &[i32],
        strides: &[i32],
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>;
}

pub struct ActiveKernel(());

impl GlobalAveragePoolKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        pads: &[i32],
        strides: &[i32],
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()> {
        rmlk_cuda::kernels::global_average_pool::compute::<f32>(
            device,
            (alpha, beta),
            pads,
            strides,
            x_data,
            x_shape,
            x_stride,
            kernel_shape,
            y_data,
            y_shape,
            y_stride,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl GlobalAveragePoolKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: T,
        _: T,
        _: &[i32],
        _: &[i32],
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &mut CudaSlice<T>,
        _: &[i32],
        _: &[i32],
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::global_average_pool::BackendHandler;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_global_average_pool_f32_2d() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 3, 3];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                    .unwrap(),
            )),
        };

        let params = TestParams {
            inputs: vec![node_a],
            attributes: vec![],
            op: Op::GlobalAveragePool,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let cuda_kernel = BackendHandler::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![5.0])
    }
}
