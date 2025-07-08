use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::binary;
#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::sub;
use rmlk_cuda::kernels::sub::SubKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct SubBackend {
    stream: Arc<CudaStream>,
}

impl SubBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl SubBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => SubKernel::SubFwdF16,
            DataType::Float => SubKernel::SubFwdF32,
            DataType::Double => SubKernel::SubFwdF64,
            DataType::Int32 => SubKernel::SubFwdI32,
            DataType::Int64 => SubKernel::SubFwdI64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        sub::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_sub<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            binary::compute::<D, D, D>("sub", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "debugger")]
        debug::write_results_binary::<D, D, D>(
            "debugging/sub",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_sub::<f16>(ctx),
            DataType::Float => self.compute_sub::<f32>(ctx),
            DataType::Double => self.compute_sub::<f64>(ctx),
            DataType::Int32 => self.compute_sub::<i32>(ctx),
            DataType::Int64 => self.compute_sub::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
