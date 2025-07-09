use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::binary;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::mul;
use rmlk_cuda::kernels::mul::MulKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct MulBackend {
    stream: Arc<CudaStream>,
}

impl MulBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl MulBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => MulKernel::FwdF16,
            DataType::Float => MulKernel::FwdF32,
            DataType::Double => MulKernel::FwdF64,
            DataType::Int32 => MulKernel::FwdI32,
            DataType::Int64 => MulKernel::FwdI64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        mul::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_mul<I>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(I::data_type())?;
        unsafe {
            binary::compute::<I, I, I>("mul", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_binary::<I, I, I>(
            "debugging/mul",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_mul::<f16>(ctx),
            DataType::Float => self.compute_mul::<f32>(ctx),
            DataType::Double => self.compute_mul::<f64>(ctx),
            DataType::Int32 => self.compute_mul::<i32>(ctx),
            DataType::Int64 => self.compute_mul::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
