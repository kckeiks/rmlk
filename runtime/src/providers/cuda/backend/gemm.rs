use crate::attributes::gemm::GemmAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cublas::StridedBatchedConfig;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use num_traits::Num;
use rmlk_cuda::kernels::gemm::GemmOp;
use rmlk_cuda::params::CudaParamMap;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct GemmBackend {
    device: Arc<CudaDevice>,
}

impl GemmBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

impl GemmBackend {
    fn compute_output_shape(&self, op: &GemmOp, ctx: &mut Context<Cuda>) -> Result<()> {
        let output_shape = op.calculate_output_shape();
        let y = ctx.get_output_mut(0)?;
        let y_shape = match y.try_shape() {
            Ok(shape) if shape.len() == 2 => &output_shape[1..],
            Ok(shape) if shape.len() == 3 => &output_shape,
            _ => {
                return Err(InternalError::InvalidTensorShape {
                    // The allocation is ok here because we're throwing an error.
                    shape: y.shape().to_vec(),
                });
            }
        };

        let y_index = y.index();
        ctx.execution_state_mut()
            .copy_from_slice(y_shape, y_index)?;

        Ok(())
    }

    fn create_gemm_op(&self, attrs: &GemmAttributes, ctx: &Context<Cuda>) -> Result<GemmOp> {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        Ok(GemmOp::new(
            &a.shape(),
            &a.stride(),
            &b.shape(),
            &b.stride(),
            attrs.trans_a(),
            attrs.trans_b(),
        ))
    }

    pub fn compute_gemm<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: CudaParamMap
            + DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + TryFrom<f32>,
        T: GemmKernel,
    {
        let attrs = GemmAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
        )?;

        let op = self.create_gemm_op(&attrs, ctx)?;

        self.compute_output_shape(&op, ctx)?;

        let alpha = D::try_from(attrs.alpha()).map_err(|_| InternalError::UnableToConvertValue)?;
        let beta = D::try_from(attrs.beta()).map_err(|_| InternalError::UnableToConvertValue)?;
        let config = op.strided_batch_config((alpha, beta))?;

        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;
        let c = ctx.get_output(0)?;

        let output_size = c.shape().iter().product();

        let a_dev_data_ref = a.try_dev_data_ptr()?;
        let a_dev_data = a_dev_data_ref.data::<D>();

        let b_dev_data_ref = b.try_dev_data_ptr()?;
        let b_dev_data = b_dev_data_ref.data::<D>();

        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let mut c = ctx.get_output_mut(0)?;
            let c_dev_data_ref = c.dev_data_ptr_mut();
            let need_to_alloc_dev_data = c_dev_data_ref.is_none()
                || c_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != output_size)
                    .unwrap_or(true);

            // We need to remove this immutable reference so we can mutate `y`.
            drop(c_dev_data_ref);

            if need_to_alloc_dev_data {
                let c_dev_data = self
                    .device
                    .alloc_zeros::<D>(output_size)
                    .map_err(rmlk_cuda::Error::from)?;
                c.set_dev_data(CudaData::new(c_dev_data));
            };
        }

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        T::execute::<D>(
            &op,
            self.device,
            &a_dev_data,
            &b_dev_data,
            &mut y_dev_data,
            config,
        )?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: GemmKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_gemm::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Conv,
                dtype,
            }),
        }
    }
}

pub trait GemmKernel {
    fn execute<T>(
        gemm_op: &GemmOp,
        device: Arc<CudaDevice>,
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        y_dev_data: &mut CudaSlice<T>,
        config: StridedBatchedConfig<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl GemmKernel for ActiveKernel {
    fn execute<T>(
        gemm_op: &GemmOp,
        device: Arc<CudaDevice>,
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        y_dev_data: &mut CudaSlice<T>,
        config: StridedBatchedConfig<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        gemm_op
            .compute::<T>(device, a_dev_data, b_dev_data, y_dev_data, config)
            .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl GemmKernel for NoOpKernel {
    fn execute<T>(
        _: &GemmOp,
        _: Arc<CudaDevice>,
        _: &CudaSlice<T>,
        _: &CudaSlice<T>,
        _: &mut CudaSlice<T>,
        _: StridedBatchedConfig<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
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
            .dev_data_ptr()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    }
}
