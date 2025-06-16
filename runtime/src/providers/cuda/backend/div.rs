use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::binary;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_cuda::kernels::div;
use rmlk_cuda::kernels::div::DivKernel;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct Divbackend {
    stream: Arc<CudaStream>,
}

impl Divbackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl Divbackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => DivKernel::DivFwdF16,
            DataType::Float => DivKernel::DivFwdF32,
            DataType::Double => DivKernel::DivFwdF64,
            DataType::Int32 => DivKernel::DivFwdI32,
            _ => {
                return Err(InternalError::UnsupportedOpForDataType {
                    op: Op::Expand,
                    dtype,
                })
            }
        };

        div::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_div<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe { binary::compute::<D, D, D>("div", self.stream, func, ctx) }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_div::<f32>(ctx),
            DataType::Int32 => self.compute_div::<i32>(ctx),
            DataType::Int64 => self.compute_div::<i64>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}
