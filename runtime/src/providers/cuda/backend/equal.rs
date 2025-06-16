use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::binary;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_cuda::kernels::equal;
use rmlk_cuda::kernels::equal::EqualKernel;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct EqualBackend {
    stream: Arc<CudaStream>,
}

impl EqualBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl EqualBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => EqualKernel::EqualFwdF16,
            DataType::Float => EqualKernel::EqualFwdF32,
            DataType::Double => EqualKernel::EqualFwdF64,
            DataType::Int32 => EqualKernel::EqualFwdI32,
            DataType::Int64 => EqualKernel::EqualFwdI64,
            _ => {
                return Err(InternalError::UnsupportedOpForDataType {
                    op: Op::Expand,
                    dtype,
                })
            }
        };

        equal::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_greater<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe { binary::compute::<D, D, bool>("equal", self.stream, func, ctx) }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_greater::<f32>(ctx),
            DataType::Int64 => self.compute_greater::<i64>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Equal,
                dtype,
            }),
        }
    }
}
