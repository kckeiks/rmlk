use crate::core::error::UnsupportedDataType;
use crate::core::Context;
use crate::providers::cuda::backend::unary;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::cos;
use rmlk_cuda::kernels::cos::CosKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct CosBackend {
    stream: Arc<CudaStream>,
}

impl CosBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl CosBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => CosKernel::CosFwdF16,
            DataType::Float => CosKernel::CosFwdF32,
            DataType::Double => CosKernel::CosFwdF64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        cos::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_cos<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            unary::compute::<D>("cos", self.stream.clone(), func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_unary::<D, D>(
            "debugging/cos",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_cos::<f32>(ctx),
            DataType::Int32 => self.compute_cos::<i32>(ctx),
            DataType::Int64 => self.compute_cos::<i64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}
