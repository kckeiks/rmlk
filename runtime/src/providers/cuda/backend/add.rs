use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::binary;
#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::add;
use rmlk_cuda::kernels::add::AddKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct AdditionBackend {
    stream: Arc<CudaStream>,
}

impl AdditionBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl AdditionBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => AddKernel::FwdF16,
            DataType::Float => AddKernel::FwdF32,
            DataType::Double => AddKernel::FwdF64,
            DataType::Int32 => AddKernel::FwdI32,
            DataType::Int64 => AddKernel::FwdI64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        add::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_addition<I>(self, ctx: &Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(I::data_type())?;
        unsafe {
            binary::compute::<I, I, I>("add", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "debugger")]
        debug::write_results_binary::<I, I, I>(
            "debugging/add",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
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
