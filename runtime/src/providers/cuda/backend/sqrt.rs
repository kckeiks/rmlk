use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::unary;
use crate::providers::cuda::backend::unary::UnaryKernel;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct SqrtBackend {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
}

impl SqrtBackend {
    pub fn new(device: &Arc<CudaDevice>, kernel: CudaFunction) -> Self {
        Self {
            device: device.clone(),
            kernel,
        }
    }

    fn compute_sqrt<I, K>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: UnaryKernel,
    {
        unsafe { unary::compute::<I, K>("sqrt", self.device.clone(), self.kernel, ctx) }
    }

    pub fn compute<K>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        K: UnaryKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_sqrt::<f32, K>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Sqrt,
                dtype,
            }),
        }
    }
}

pub struct ActiveKernel(());

impl UnaryKernel for ActiveKernel {
    fn execute<T>(
        kernel: CudaFunction,
        input_dev_data: &CudaSlice<T>,
        output_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        unsafe {
            rmlk_cuda::kernels::unary::compute(kernel, input_dev_data, output_dev_data)?;
        }

        Ok(())
    }
}
