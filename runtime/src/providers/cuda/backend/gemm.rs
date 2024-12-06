use crate::attributes::gemm::GemmAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cublas::StridedBatchedConfig;
use cudarc::driver::{CudaDevice, CudaSlice};
use rmlk_cuda::kernels::gemm::GemmOp;
use rmlk_schema::{DataType, Op};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct GemmBackend<T> {
    device: Arc<CudaDevice>,
    _marker: PhantomData<T>,
}

impl<T> GemmBackend<T>
where
    T: GemmKernel,
{
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            _marker: PhantomData,
        }
    }
}

impl<T> GemmBackend<T>
where
    T: GemmKernel,
{
    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        let attrs = GemmAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
        )?;

        let op = GemmOp::new(
            &a.shape(),
            &a.stride(),
            &b.shape(),
            &b.stride(),
            attrs.trans_a(),
            attrs.trans_b(),
        );

        let output_shape = op.calculate_output_shape();
        let output_size = output_shape.iter().product();

        let dev_data =
            if matches!(a.dtype(), DataType::Float) {
                let mut y_dev_data = self
                    .device
                    .alloc_zeros(output_size)
                    .map_err(rmlk_cuda::Error::from)?;

                let config = op.strided_batch_config((attrs.alpha(), attrs.beta()))?;

                let a_dev_data_ref = a.dev_data().ok_or(InternalError::MissingDeviceData)?;
                let a_dev_data = a_dev_data_ref.f32().ok_or_else(|| {
                    InternalError::UnexpectedTensorDataType {
                        expected: DataType::Float,
                    }
                })?;

                let b_dev_data_ref = b.dev_data().ok_or(InternalError::MissingDeviceData)?;
                let b_dev_data = b_dev_data_ref.f32().ok_or_else(|| {
                    InternalError::UnexpectedTensorDataType {
                        expected: DataType::Float,
                    }
                })?;

                T::execute_with_float_tensors(
                    &op,
                    self.device,
                    a_dev_data,
                    b_dev_data,
                    &mut y_dev_data,
                    config,
                )?;

                CudaData::F32(y_dev_data)
            } else {
                return Err(InternalError::UnsupportedOpForDataType {
                    op: Op::Gemm,
                    dtype: *a.dtype(),
                });
            };

        let y_dtype = *a.dtype();
        let mut y = ctx.get_output_mut(0)?;
        y.reshape(output_shape.as_slice())?;
        y.set_dev_data(dev_data);
        y.set_dtype(y_dtype);

        Ok(())
    }
}

pub trait GemmKernel {
    fn execute_with_float_tensors(
        // Todo: refactor this API.
        gemm_op: &GemmOp,
        device: Arc<CudaDevice>,
        a_dev_data: &CudaSlice<f32>,
        b_dev_data: &CudaSlice<f32>,
        y_dev_data: &mut CudaSlice<f32>,
        config: StridedBatchedConfig<f32>,
    ) -> Result<()>;
}

pub struct ActiveKernel(());

impl GemmKernel for ActiveKernel {
    fn execute_with_float_tensors(
        gemm_op: &GemmOp,
        device: Arc<CudaDevice>,
        a_dev_data: &CudaSlice<f32>,
        b_dev_data: &CudaSlice<f32>,
        y_dev_data: &mut CudaSlice<f32>,
        config: StridedBatchedConfig<f32>,
    ) -> Result<()> {
        gemm_op
            .compute_f32(device, a_dev_data, b_dev_data, y_dev_data, config)
            .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl GemmKernel for NoOpKernel {
    fn execute_with_float_tensors(
        _: &GemmOp,
        _: Arc<CudaDevice>,
        _: &CudaSlice<f32>,
        _: &CudaSlice<f32>,
        _: &mut CudaSlice<f32>,
        _: StridedBatchedConfig<f32>,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::gemm::BackendHandler;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_gemm_f32() {
        let device = CudaDevice::new(0).unwrap();

        let shape = vec![1, 2, 2];
        let dtype = DataType::Float;
        let op = Op::Gemm;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            attributes: vec![],
            op,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 3).unwrap();

        let cuda_kernel = BackendHandler::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .dev_data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    }
}
