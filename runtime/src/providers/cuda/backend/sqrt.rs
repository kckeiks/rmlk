use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::unary;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
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
            DataType::Float => self.compute_sqrt::<f32>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Sqrt,
                dtype,
            }),
        }
    }
}
