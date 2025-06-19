use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::binary;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct AdditionBackend {
    stream: Arc<CudaStream>,
    f: CudaFunction,
}

impl AdditionBackend {
    pub fn new(stream: Arc<CudaStream>, f: CudaFunction) -> Self {
        Self { stream, f }
    }
}

impl AdditionBackend {
    fn compute_addition<I>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        unsafe { binary::compute::<I, I, I>("add", self.stream, self.f, ctx) }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_addition::<f16>(ctx),
            DataType::Float => self.compute_addition::<f32>(ctx),
            DataType::Double => self.compute_addition::<f64>(ctx),
            DataType::Int32 => self.compute_addition::<i32>(ctx),
            DataType::Uint32 => self.compute_addition::<u32>(ctx),
            DataType::Int64 => self.compute_addition::<i64>(ctx),
            DataType::Uint64 => self.compute_addition::<u64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
