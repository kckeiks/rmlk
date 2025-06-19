use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::unary;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct SqrtBackend {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
}

impl SqrtBackend {
    pub fn new(stream: &Arc<CudaStream>, kernel: CudaFunction) -> Self {
        Self {
            stream: stream.clone(),
            kernel,
        }
    }

    fn compute_sqrt<I>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        unsafe { unary::compute::<I>("sqrt", self.stream.clone(), self.kernel, ctx) }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_sqrt::<f16>(ctx),
            DataType::Float => self.compute_sqrt::<f32>(ctx),
            DataType::Double => self.compute_sqrt::<f64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
