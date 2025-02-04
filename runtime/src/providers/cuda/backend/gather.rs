use crate::attributes;
use crate::core::allocators::ScratchAllocator;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct GatherBackend {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl GatherBackend {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
    }
}

impl GatherBackend {
    fn compute_gather<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: GatherKernel,
    {
        let axis = ctx
            .get_attributes()
            .map(attributes::gather::get_axis)
            .unwrap_or(0);

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: GatherKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_gather::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}

trait GatherKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        alloc: &ScratchAllocator,
        a_dev_data: &CudaSlice<T>,
        a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        b_shape: &[usize],
        b_stride: &[usize],
        c_shape: &[usize],
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl GatherKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        alloc: &ScratchAllocator,
        a_dev_data: &CudaSlice<T>,
        _a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        _b_shape: &[usize],
        b_stride: &[usize],
        c_shape: &[usize],
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        todo!()
    }
}

pub struct NoOpKernel(());

impl GatherKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: CudaFunction,
        _: &ScratchAllocator,
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &[usize],
        _: &mut CudaSlice<T>,
    ) -> Result<()> {
        Ok(())
    }
}
