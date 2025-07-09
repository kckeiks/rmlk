use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::unary;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::sqrt::SqrtKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct SqrtBackend {
    stream: Arc<CudaStream>,
}

impl SqrtBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => SqrtKernel::FwdF16,
            DataType::Float => SqrtKernel::FwdF32,
            DataType::Double => SqrtKernel::FwdF64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        rmlk_cuda::kernels::sqrt::load_kernel(self.stream.context().clone(), kernel_name)
            .map_err(Into::into)
    }

    fn compute_sqrt<I>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let kernel = self.load_cuda_function(I::data_type())?;
        unsafe {
            unary::compute::<I>("sqrt", self.stream.clone(), kernel, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_unary::<I, I>(
            "debugging/sqrt",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
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
