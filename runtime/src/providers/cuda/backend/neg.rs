use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::unary;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_cuda::kernels::neg;
use rmlk_cuda::kernels::neg::NegKernel;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct NegBackend {
    stream: Arc<CudaStream>,
}

impl NegBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl NegBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => NegKernel::NegFwdF16,
            DataType::Float => NegKernel::NegFwdF32,
            DataType::Double => NegKernel::NegFwdF64,
            DataType::Int32 => NegKernel::NegFwdI32,
            _ => {
                return Err(InternalError::UnsupportedDataTypeForOp {
                    op: Op::Expand,
                    dtype,
                })
            }
        };

        neg::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_div<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe { unary::compute::<D>("neg", self.stream, func, ctx) }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_div::<f32>(ctx),
            DataType::Int32 => self.compute_div::<i32>(ctx),
            DataType::Int64 => self.compute_div::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataTypeForOp { op: Op::Neg, dtype }),
        }
    }
}
