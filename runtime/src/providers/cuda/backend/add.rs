use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, ValidAsZeroBits};
use rmlk_schema::{DataType, Op};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct AdditionBackend<T> {
    device: Arc<CudaDevice>,
    f: CudaFunction,
    _marker: PhantomData<T>,
}

impl<T> AdditionBackend<T>
where
    T: AdditionKernel,
{
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self {
            device,
            f,
            _marker: PhantomData,
        }
    }
}

impl<T> AdditionBackend<T>
where
    T: AdditionKernel,
{
    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        let elem_count: usize = a.shape().iter().product();

        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let info_buffer = scratch_alloc.allocate(3 * a.shape().len())?;

        let dev_data =
            if matches!(a.dtype(), DataType::Float) {
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

                let mut c_dev_data = unsafe {
                    self.device
                        .alloc::<f32>(elem_count)
                        .map_err(rmlk_cuda::Error::from)?
                };

                T::execute::<f32>(
                    self.device,
                    self.f,
                    a_dev_data,
                    &a.shape(),
                    &a.stride(),
                    b_dev_data,
                    &b.shape(),
                    &b.stride(),
                    &mut c_dev_data,
                    info_buffer,
                )?;

                CudaData::F32(c_dev_data)
            } else {
                return Err(InternalError::UnsupportedOpForDataType {
                    op: Op::Add,
                    dtype: *a.dtype(),
                });
            };

        let dtype = *a.dtype();
        let mut c = ctx.get_output_mut(0)?;
        c.reshape(&a.shape())?;
        c.set_dev_data(dev_data);
        c.set_dtype(dtype);

        Ok(())
    }
}

pub trait AdditionKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        lhs_data: &CudaSlice<T>,
        lhs_shape: &[usize],
        lhs_stride: &[usize],
        rhs_data: &CudaSlice<T>,
        rhs_shape: &[usize],
        rhs_stride: &[usize],
        out_data: &mut CudaSlice<T>,
        info_buffer: &mut [usize],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl AdditionKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        lhs_data: &CudaSlice<T>,
        lhs_shape: &[usize],
        lhs_stride: &[usize],
        rhs_data: &CudaSlice<T>,
        rhs_shape: &[usize],
        rhs_stride: &[usize],
        out_data: &mut CudaSlice<T>,
        info_buffer: &mut [usize],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::add::compute(
            device,
            func,
            lhs_data,
            lhs_shape,
            lhs_stride,
            rhs_data,
            rhs_shape,
            rhs_stride,
            out_data,
            info_buffer,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl AdditionKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: CudaFunction,
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &mut CudaSlice<T>,
        _: &mut [usize],
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::add::BackendHandler;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_add_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![4, 1, 1, 1];
        let dtype = DataType::Float;

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
            op: Op::Add,
            attributes: vec![],
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 3).unwrap();

        let f = rmlk_cuda::load_kernel(&device, Op::Add, DataType::Float).unwrap();
        let cuda_kernel = BackendHandler::new(device.clone(), f);
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .dev_data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }
}
