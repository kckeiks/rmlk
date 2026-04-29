use crate::core::error::UnsupportedDataType;
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
use rmlk_cuda::kernels::sin;
use rmlk_cuda::kernels::sin::SinKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct SinBackend {
    stream: Arc<CudaStream>,
}

impl SinBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl SinBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => SinKernel::SinFwdF16,
            DataType::Float => SinKernel::SinFwdF32,
            DataType::Double => SinKernel::SinFwdF64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        sin::load_kernel(self.stream.context().clone(), kernel_name)
            .map_err(Box::new)
            .map_err(Into::into)
    }

    fn compute_sin<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            unary::compute::<D>("sin", self.stream.clone(), func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_unary::<D, D>(
            "debugging/sin",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_sin::<f16>(ctx),
            DataType::Float => self.compute_sin::<f32>(ctx),
            DataType::Double => self.compute_sin::<f64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}
