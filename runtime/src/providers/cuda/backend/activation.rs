use crate::core::device_service::DeviceService;
use crate::core::error::{InternalError, Result};
use crate::core::{ScratchAllocator, Tensor};
use crate::ops::activation::ActivationBackend;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
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
    T: ActivationKernel,
{
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            _marker: PhantomData,
        }
    }
}

impl<T> ActivationBackend for BackendHandler<T>
where
    T: ActivationKernel,
{
    type Service = Cuda;
    fn compute(
        &self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        scratch_alloc: &ScratchAllocator,
    ) -> Result<<Self::Service as DeviceService>::Data> {
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x.data().and_then(|data| data.f32()).ok_or_else(|| {
                InternalError::UnexpectedTensorDataType {
                    expected: DataType::Float,
                }
            })?;

            let mut y_data = self
                .device
                .alloc_zeros::<f32>(x.shape().iter().copied().product::<usize>())
                .map_err(rmlk_cuda::Error::from)?;

            T::execute::<f32>(
                self.device.clone(),
                1.0,
                0.0,
                x_data,
                x_shape,
                x_stride,
                &mut y_data,
            )?;

            Ok(CudaData::F32(y_data))
        } else {
            // Todo: Update op.
            Err(InternalError::UnsupportedOpForDataType {
                op: Op::Relu,
                dtype: *x.dtype(),
            })
        }
    }
}

trait ActivationKernel {
    fn execute<T: CudnnDataType>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_data: &mut CudaSlice<T>,
    ) -> Result<()>;
}

pub struct ActiveKernel(());

impl ActivationKernel for ActiveKernel {
    fn execute<T: CudnnDataType>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_data: &mut CudaSlice<T>,
    ) -> Result<()> {
        rmlk_cuda::kernels::activation::compute(
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

pub struct NoOpKernel(());

impl ActivationKernel for NoOpKernel {
    fn execute<T: CudnnDataType>(
        _: Arc<CudaDevice>,
        _: T,
        _: T,
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &mut CudaSlice<T>,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::activation::ActivationOp;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_relu_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 2, 2];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![-1.0, 2.0, -3.0, 100.0]).unwrap(),
            )),
        };

        let params = TestParams {
            inputs: vec![node_a],
            attributes: Vec::new(),
            op: Op::Relu,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let cuda_kernel = ActivationOp::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![0.0, 2.0, 0.0, 100.0])
    }
}
