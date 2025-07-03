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
use rmlk_cuda::kernels::greater;
use rmlk_cuda::kernels::greater::GreaterKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct GreaterBackend {
    stream: Arc<CudaStream>,
}

impl GreaterBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl GreaterBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => GreaterKernel::GreaterFwdF16,
            DataType::Float => GreaterKernel::GreaterFwdF32,
            DataType::Double => GreaterKernel::GreaterFwdF64,
            DataType::Int32 => GreaterKernel::GreaterFwdI32,
            DataType::Int64 => GreaterKernel::GreaterFwdI64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        greater::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_greater<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            binary::compute::<D, D, bool>("greater", self.stream.clone(), func, ctx)?;
        }
        #[cfg(feature = "debugger")]
        debug::write_results_binary::<D, D, bool>(
            "debugging/greater",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        /*self.stream
        .synchronize()
        .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_greater::<f16>(ctx),
            DataType::Float => self.compute_greater::<f32>(ctx),
            DataType::Double => self.compute_greater::<f64>(ctx),
            DataType::Int32 => self.compute_greater::<i32>(ctx),
            DataType::Int64 => self.compute_greater::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
