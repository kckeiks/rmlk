use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use rmlk_schema::{DataType, Op};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct ActivationBackend<T> {
    device: Arc<CudaDevice>,
    _marker: PhantomData<T>,
}

impl<T> ActivationBackend<T>
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

impl<T> ActivationBackend<T>
where
    T: ActivationKernel,
{
    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;

        // Check type of all inputs and outputs here.
        if matches!(x.dtype(), DataType::Float) {
            let x_dev_data_ref = x.dev_data().ok_or(InternalError::MissingDeviceData)?;
            let x_dev_data =
                x_dev_data_ref
                    .f32()
                    .ok_or_else(|| InternalError::UnexpectedTensorDataType {
                        expected: DataType::Float,
                    })?;

            // Allocate device data for the tensor if we haven't done it yet
            // or if the existing allocated data has a different size.
            {
                let expected_len = x.shape().iter().product();
                let mut y = ctx.get_output_mut(0)?;
                let y_dev_data_ref = y.dev_data_mut();
                let need_to_alloc_dev_data = y_dev_data_ref.is_none()
                    || y_dev_data_ref
                        .as_ref()
                        .and_then(|data| data.f32().map(|data| data.len() != expected_len))
                        .unwrap_or(true);

                // We need to remove this immutable reference so we can mutate `y`.
                drop(y_dev_data_ref);

                if need_to_alloc_dev_data {
                    let y_dev_data = self
                        .device
                        .alloc_zeros::<f32>(x.shape().iter().copied().product::<usize>())
                        .map_err(rmlk_cuda::Error::from)?;
                    y.set_dev_data(CudaData::F32(y_dev_data));
                };
            }

            // The device data should exist so we will execute the kernel
            // and update the destination device data with the result.
            let y = ctx.get_output_mut(0)?;
            let mut y_dev_data_ref = y.dev_data_mut();
            let y_dev_data = y_dev_data_ref
                .as_mut()
                .expect("we already checked that it initialized")
                ._f32_mut()
                .ok_or_else(|| InternalError::UnexpectedTensorDataType {
                    expected: DataType::Float,
                })?;

            T::execute::<f32>(
                self.device.clone(),
                1.0,
                0.0,
                x_dev_data,
                x_shape,
                x_stride,
                y_dev_data,
            )?;
        } else {
            // Todo: Update op.
            return Err(InternalError::UnsupportedOpForDataType {
                op: Op::Relu,
                dtype: *x.dtype(),
            });
        };

        let dtype = *x.dtype();
        let mut y = ctx.get_output_mut(0)?;
        y.reshape(&x.shape())?;
        y.set_dtype(dtype);

        Ok(())
    }
}

pub trait ActivationKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_dev_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl ActivationKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
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
            .dev_data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![0.0, 2.0, 0.0, 100.0])
    }
}
