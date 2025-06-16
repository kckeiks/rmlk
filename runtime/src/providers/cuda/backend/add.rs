use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::binary;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
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
    fn compute_addition<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        unsafe { binary::compute::<D, D, D>("add", self.stream, self.f, ctx) }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_addition::<f32>(ctx),
            DataType::Int64 => self.compute_addition::<i64>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}
