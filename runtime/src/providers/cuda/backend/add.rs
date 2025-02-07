use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::binary;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct AdditionBackend {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl AdditionBackend {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
    }
}

impl AdditionBackend {
    fn compute_addition<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: AdditionKernel,
    {
        unsafe { binary::compute::<D, T>("add", self.device, self.f, ctx) }
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: AdditionKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_addition::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}

pub trait AdditionKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        rank: usize,
        info: &[usize],
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl AdditionKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        rank: usize,
        info: &[usize],
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        unsafe {
            rmlk_cuda::kernels::binary::compute(
                device, func, rank, info, a_dev_data, b_dev_data, c_dev_data,
            )
            .map_err(Into::into)
        }
    }
}

pub struct NoOpKernel(());

impl AdditionKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: CudaFunction,
        _: usize,
        _: &[usize],
        _: &CudaSlice<T>,
        _: &CudaSlice<T>,
        _: &mut CudaSlice<T>,
    ) -> Result<()> {
        Ok(())
    }
}
