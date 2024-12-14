use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct ActivationBackend {
    device: Arc<CudaDevice>,
}

impl ActivationBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

impl ActivationBackend {
    fn compute_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;
        let x_index = x.src_id();
        let y_index = y.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_within(x_index, y_index)
    }

    fn compute_activation<D, T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: ActivationKernel,
    {
        self.compute_output_shape(ctx)?;

        let x = ctx.get_input(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;

        let x_dev_data_ref = x.try_dev_data_ptr()?;
        let x_dev_data = x_dev_data_ref.data::<D>();

        // Todo: would it help readability to put this in a func?
        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let expected_len = x.shape().iter().product();
            let mut y = ctx.get_output(0)?;
            let y_dev_data_ref = y.dev_data_ptr_mut();
            let need_to_alloc_dev_data = y_dev_data_ref.is_none()
                || y_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != expected_len)
                    .unwrap_or(true);

            // We need to remove this immutable reference so we can mutate `y`.
            drop(y_dev_data_ref);

            if need_to_alloc_dev_data {
                let y_dev_data = self
                    .device
                    .alloc_zeros::<D>(x.shape().iter().copied().product::<usize>())
                    .map_err(rmlk_cuda::Error::from)?;
                y.set_dev_data(CudaData::new(y_dev_data));
            };
        }

        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        T::execute::<D>(
            self.device.clone(),
            D::one(),
            D::zero(),
            &x_dev_data,
            x_shape,
            x_stride,
            &mut y_dev_data,
        )?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: ActivationKernel,
    {
        // Todo: we need to add validation to make sure the tensor types meets
        // the expected data type.
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_activation::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Relu,
                dtype,
            }),
        }
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
            .dev_data_ptr()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![0.0, 2.0, 0.0, 100.0])
    }
}
