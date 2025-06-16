use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::binary;
use crate::providers::cuda::backend::binary::BinaryKernel;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_cuda::kernels::greater::GreaterKernel;
use rmlk_cuda::kernels::{greater};
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct GreaterBackend {
    stream: Arc<CudaStream>,
}

impl GreaterBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
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
            _ => {
                return Err(InternalError::UnsupportedOpForDataType {
                    op: Op::Expand,
                    dtype,
                })
            }
        };

        greater::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_greater<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe { binary::compute::<D, GreaterFunc>("greater", self.stream, func, ctx) }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()>
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_greater::<f32>(ctx),
            DataType::Int64 => self.compute_greater::<i64>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}

pub struct GreaterFunc(());

impl BinaryKernel for GreaterFunc {
    fn execute<T>(
        stream: Arc<CudaStream>,
        func: CudaFunction,
        rank: usize,
        info: &[usize],
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        unsafe {
            rmlk_cuda::kernels::binary::compute(
                stream, func, rank, info, a_dev_data, b_dev_data, c_dev_data,
            )
                .map_err(Into::into)
        }
    }
}