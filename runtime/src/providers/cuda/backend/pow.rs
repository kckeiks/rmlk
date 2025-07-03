use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::binary;
#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::pow;
use rmlk_cuda::kernels::pow::PowKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct PowBackend {
    stream: Arc<CudaStream>,
}

impl PowBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl PowBackend {
    fn load_cuda_function(
        &self,
        base_dtype: DataType,
        power_dtype: DataType,
    ) -> Result<CudaFunction> {
        let kernel_name = match (base_dtype, power_dtype) {
            (DataType::Float16, DataType::Uint8) => PowKernel::PowFwdF16U8,
            (DataType::Float16, DataType::Uint16) => PowKernel::PowFwdF16U16,
            (DataType::Float16, DataType::Uint32) => PowKernel::PowFwdF16U32,
            (DataType::Float16, DataType::Uint64) => PowKernel::PowFwdF16U64,
            (DataType::Float16, DataType::Int8) => PowKernel::PowFwdF16I8,
            (DataType::Float16, DataType::Int16) => PowKernel::PowFwdF16I16,
            (DataType::Float16, DataType::Int32) => PowKernel::PowFwdF16I32,
            (DataType::Float16, DataType::Int64) => PowKernel::PowFwdF16I64,
            (DataType::Float16, DataType::Float16) => PowKernel::PowFwdF16F16,
            (DataType::Float16, DataType::Float) => PowKernel::PowFwdF16F32,
            (DataType::Float16, DataType::Double) => PowKernel::PowFwdF16F64,
            (DataType::Float, DataType::Uint8) => PowKernel::PowFwdF32U8,
            (DataType::Float, DataType::Uint16) => PowKernel::PowFwdF32U16,
            (DataType::Float, DataType::Uint32) => PowKernel::PowFwdF32U32,
            (DataType::Float, DataType::Uint64) => PowKernel::PowFwdF32U64,
            (DataType::Float, DataType::Int8) => PowKernel::PowFwdF32I8,
            (DataType::Float, DataType::Int16) => PowKernel::PowFwdF32I16,
            (DataType::Float, DataType::Int32) => PowKernel::PowFwdF32I32,
            (DataType::Float, DataType::Int64) => PowKernel::PowFwdF32I64,
            (DataType::Float, DataType::Float16) => PowKernel::PowFwdF32F16,
            (DataType::Float, DataType::Float) => PowKernel::PowFwdF32F32,
            (DataType::Float, DataType::Double) => PowKernel::PowFwdF32F64,
            (DataType::Double, DataType::Uint8) => PowKernel::PowFwdF64U8,
            (DataType::Double, DataType::Uint16) => PowKernel::PowFwdF64U16,
            (DataType::Double, DataType::Uint32) => PowKernel::PowFwdF64U32,
            (DataType::Double, DataType::Uint64) => PowKernel::PowFwdF64U64,
            (DataType::Double, DataType::Int8) => PowKernel::PowFwdF64I8,
            (DataType::Double, DataType::Int16) => PowKernel::PowFwdF64I16,
            (DataType::Double, DataType::Int32) => PowKernel::PowFwdF64I32,
            (DataType::Double, DataType::Int64) => PowKernel::PowFwdF64I64,
            (DataType::Double, DataType::Float16) => PowKernel::PowFwdF64F16,
            (DataType::Double, DataType::Float) => PowKernel::PowFwdF64F32,
            (DataType::Double, DataType::Double) => PowKernel::PowFwdF64F64,
            (DataType::Int32, DataType::Uint8) => PowKernel::PowFwdI32U8,
            (DataType::Int32, DataType::Uint16) => PowKernel::PowFwdI32U16,
            (DataType::Int32, DataType::Uint32) => PowKernel::PowFwdI32U32,
            (DataType::Int32, DataType::Uint64) => PowKernel::PowFwdI32U64,
            (DataType::Int32, DataType::Int8) => PowKernel::PowFwdI32I8,
            (DataType::Int32, DataType::Int16) => PowKernel::PowFwdI32I16,
            (DataType::Int32, DataType::Int32) => PowKernel::PowFwdI32I32,
            (DataType::Int32, DataType::Int64) => PowKernel::PowFwdI32I64,
            (DataType::Int32, DataType::Float16) => PowKernel::PowFwdI32F16,
            (DataType::Int32, DataType::Float) => PowKernel::PowFwdI32F32,
            (DataType::Int32, DataType::Double) => PowKernel::PowFwdI32F64,
            (DataType::Int64, DataType::Uint8) => PowKernel::PowFwdI64U8,
            (DataType::Int64, DataType::Uint16) => PowKernel::PowFwdI64U16,
            (DataType::Int64, DataType::Uint32) => PowKernel::PowFwdI64U32,
            (DataType::Int64, DataType::Uint64) => PowKernel::PowFwdI64U64,
            (DataType::Int64, DataType::Int8) => PowKernel::PowFwdI64I8,
            (DataType::Int64, DataType::Int16) => PowKernel::PowFwdI64I16,
            (DataType::Int64, DataType::Int32) => PowKernel::PowFwdI64I32,
            (DataType::Int64, DataType::Int64) => PowKernel::PowFwdI64I64,
            (DataType::Int64, DataType::Float16) => PowKernel::PowFwdI64F16,
            (DataType::Int64, DataType::Float) => PowKernel::PowFwdI64F32,
            (DataType::Int64, DataType::Double) => PowKernel::PowFwdI64F64,
            (_, dtype2) => return Err(InternalError::UnsupportedDataType { dtype: dtype2 }.into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        pow::load_kernel(&self.stream.context(), kernel_name).map_err(Into::into)
    }

    fn compute_pow<X, Y>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        X: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        Y: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(X::data_type(), Y::data_type())?;

        unsafe {
            binary::compute::<X, Y, X>("pow", self.stream.clone(), func, ctx)?;
        }

        #[cfg(feature = "debugger")]
        debug::write_results_binary::<X, Y, X>(
            "debugging/pow",
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
        let base_dtype = ctx.get_input(0)?.dtype();
        let power_dtype = ctx.get_input(1)?.dtype();

        match (base_dtype, power_dtype) {
            (DataType::Int64, DataType::Int32) => self.compute_pow::<i64, i32>(ctx),
            (DataType::Int64, DataType::Int64) => self.compute_pow::<i64, i64>(ctx),
            (DataType::Int64, DataType::Float) => self.compute_pow::<i64, f32>(ctx),
            (DataType::Float, DataType::Int32) => self.compute_pow::<f32, i32>(ctx),
            (DataType::Float, DataType::Int64) => self.compute_pow::<f32, i64>(ctx),
            (DataType::Float, DataType::Float) => self.compute_pow::<f32, f32>(ctx),
            (DataType::Int32, DataType::Int32) => self.compute_pow::<i32, i32>(ctx),
            (DataType::Int32, DataType::Int64) => self.compute_pow::<i32, i64>(ctx),
            (DataType::Int32, DataType::Float) => self.compute_pow::<i32, f32>(ctx),
            // (DataType::Int64, DataType::Float16) => self.compute_pow::<i64, f16>(ctx),
            // (DataType::Float, DataType::Float16) => self.compute_pow::<f32, f16>(ctx),
            // (DataType::Int32, DataType::Float16) => self.compute_pow::<i32, f16>(ctx),
            // (DataType::Float16, DataType::Int64) => self.compute_pow::<f16, i64>(ctx),
            // (DataType::Float16, DataType::Int32) => self.compute_pow::<f16, i32>(ctx),
            // (DataType::Float16, DataType::Float) => self.compute_pow::<f16, f32>(ctx),
            // (DataType::Float16, DataType::Float16) => self.compute_pow::<f16, f16>(ctx),
            // (DataType::Float16, DataType::Double) => self.compute_pow::<f16, f64>(ctx),
            // (DataType::Float, DataType::Uint8) => self.compute_pow::<f32, u8>(ctx),
            // (DataType::Float, DataType::Uint16) => self.compute_pow::<f32, u16>(ctx),
            // (DataType::Float, DataType::Uint32) => self.compute_pow::<f32, u32>(ctx),
            // (DataType::Float, DataType::Uint64) => self.compute_pow::<f32, u64>(ctx),
            // (DataType::Float, DataType::Int8) => self.compute_pow::<f32, i8>(ctx),
            // (DataType::Float, DataType::Int16) => self.compute_pow::<f32, i16>(ctx),
            // (DataType::Float, DataType::Double) => self.compute_pow::<f32, f64>(ctx),
            // (DataType::Double, DataType::Uint8) => self.compute_pow::<f64, u8>(ctx),
            // (DataType::Double, DataType::Uint16) => self.compute_pow::<f64, u16>(ctx),
            // (DataType::Double, DataType::Uint32) => self.compute_pow::<f64, u32>(ctx),
            // (DataType::Double, DataType::Uint64) => self.compute_pow::<f64, u64>(ctx),
            // (DataType::Double, DataType::Int8) => self.compute_pow::<f64, i8>(ctx),
            // (DataType::Double, DataType::Int16) => self.compute_pow::<f64, i16>(ctx),
            // (DataType::Double, DataType::Int32) => self.compute_pow::<f64, i32>(ctx),
            // (DataType::Double, DataType::Int64) => self.compute_pow::<f64, i64>(ctx),
            // (DataType::Double, DataType::Float16) => self.compute_pow::<f64, f16>(ctx),
            // (DataType::Double, DataType::Float) => self.compute_pow::<f64, f32>(ctx),
            // (DataType::Double, DataType::Double) => self.compute_pow::<f64, f64>(ctx),
            // (DataType::Int32, DataType::Uint8) => self.compute_pow::<i32, u8>(ctx),
            // (DataType::Int32, DataType::Uint16) => self.compute_pow::<i32, u16>(ctx),
            // (DataType::Int32, DataType::Uint32) => self.compute_pow::<i32, u32>(ctx),
            // (DataType::Int32, DataType::Uint64) => self.compute_pow::<i32, u64>(ctx),
            // (DataType::Int32, DataType::Int8) => self.compute_pow::<i32, i8>(ctx),
            // (DataType::Int32, DataType::Int16) => self.compute_pow::<i32, i16>(ctx),
            // (DataType::Int32, DataType::Double) => self.compute_pow::<i32, f64>(ctx),
            // (DataType::Int64, DataType::Uint8) => self.compute_pow::<i64, u8>(ctx),
            // (DataType::Int64, DataType::Uint16) => self.compute_pow::<i64, u16>(ctx),
            // (DataType::Int64, DataType::Uint32) => self.compute_pow::<i64, u32>(ctx),
            // (DataType::Int64, DataType::Uint64) => self.compute_pow::<i64, u64>(ctx),
            // (DataType::Int64, DataType::Int8) => self.compute_pow::<i64, i8>(ctx),
            // (DataType::Int64, DataType::Int16) => self.compute_pow::<i64, i16>(ctx),
            // (DataType::Int64, DataType::Double) => self.compute_pow::<i64, f64>(ctx),
            // (DataType::Float16, DataType::Uint8) => self.compute_pow::<f16, u8>(ctx),
            // (DataType::Float16, DataType::Uint16) => self.compute_pow::<f16, u16>(ctx),
            // (DataType::Float16, DataType::Uint32) => self.compute_pow::<f16, u32>(ctx),
            // (DataType::Float16, DataType::Uint64) => self.compute_pow::<f16, u64>(ctx),
            // (DataType::Float16, DataType::Int8) => self.compute_pow::<f16, i8>(ctx),
            // (DataType::Float16, DataType::Int16) => self.compute_pow::<f16, i16>(ctx),
            (_, dtype2) => Err(InternalError::UnsupportedDataType { dtype: dtype2 }.into()),
        }
    }
}
