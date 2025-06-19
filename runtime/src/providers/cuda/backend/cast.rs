use crate::attributes::cast;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::cast::CastKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct CastBackend {
    stream: Arc<CudaStream>,
}

impl CastBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, src: DataType, dst: DataType) -> Result<CudaFunction> {
        let kernel = match (src, dst) {
            (DataType::Float16, DataType::Float) => CastKernel::F16ToF32,
            (DataType::Float16, DataType::Double) => CastKernel::F16ToF64,
            (DataType::Float16, DataType::Int32) => CastKernel::F16ToI32,
            (DataType::Float16, DataType::Uint32) => CastKernel::F16ToU32,
            (DataType::Float16, DataType::Int64) => CastKernel::F16ToI64,
            (DataType::Float16, DataType::Uint64) => CastKernel::F16ToU64,
            (DataType::Float, DataType::Float16) => CastKernel::F32ToF16,
            (DataType::Float, DataType::Double) => CastKernel::F32ToF64,
            (DataType::Float, DataType::Int32) => CastKernel::F32ToI32,
            (DataType::Float, DataType::Uint32) => CastKernel::F32ToU32,
            (DataType::Float, DataType::Int64) => CastKernel::F32ToI64,
            (DataType::Float, DataType::Uint64) => CastKernel::F32ToU64,
            (DataType::Double, DataType::Float16) => CastKernel::F64ToF16,
            (DataType::Double, DataType::Float) => CastKernel::F64ToF32,
            (DataType::Double, DataType::Int32) => CastKernel::F64ToI32,
            (DataType::Double, DataType::Uint32) => CastKernel::F64ToU32,
            (DataType::Double, DataType::Int64) => CastKernel::F64ToI64,
            (DataType::Double, DataType::Uint64) => CastKernel::F64ToU64,
            (DataType::Int32, DataType::Float16) => CastKernel::I32ToF16,
            (DataType::Int32, DataType::Float) => CastKernel::I32ToF32,
            (DataType::Int32, DataType::Double) => CastKernel::I32ToF64,
            (DataType::Int32, DataType::Int64) => CastKernel::I32ToI64,
            (DataType::Uint32, DataType::Float16) => CastKernel::U32ToF16,
            (DataType::Uint32, DataType::Float) => CastKernel::U32ToF32,
            (DataType::Uint32, DataType::Double) => CastKernel::U32ToF64,
            (DataType::Uint32, DataType::Uint64) => CastKernel::U32ToU64,
            (DataType::Int64, DataType::Float16) => CastKernel::I64ToF16,
            (DataType::Int64, DataType::Float) => CastKernel::I64ToF32,
            (DataType::Int64, DataType::Double) => CastKernel::I64ToF64,
            (DataType::Int64, DataType::Int32) => CastKernel::I64ToI32,
            (DataType::Uint64, DataType::Float16) => CastKernel::U64ToF16,
            (DataType::Uint64, DataType::Float) => CastKernel::U64ToF32,
            (DataType::Uint64, DataType::Double) => CastKernel::U64ToF64,
            (DataType::Uint64, DataType::Uint32) => CastKernel::U64ToU32,
            _ => return Err(CastError::UnsupportedCast { src, dst }.into()),
        };

        rmlk_cuda::load_cast_kernel(self.stream.context(), kernel).map_err(Into::into)
    }

    fn compute_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let input = ctx.get_input(0)?;
        let output = ctx.get_output(0)?;
        let src_id = input.src_id();
        let dst_id = output.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_within(src_id, dst_id)?;
        Ok(())
    }

    pub fn compute_cast<I, O>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
        O: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        self.compute_output_shape(ctx)?;

        let kernel = self.load_cuda_function(I::data_type(), O::data_type())?;

        let output_tensor = ctx.get_output(0)?;

        debug!(
            "[output][shape={:?}][stride=[{:?}]",
            output_tensor.shape(),
            output_tensor.stride()
        );

        common::init_tensor_device_data::<O>(&self.stream, output_tensor)?;

        let input = ctx.get_input(0)?;

        debug!(
            "[input][shape={:?}][stride=[{:?}]",
            input.shape(),
            input.stride()
        );

        let input_dev_data_ref = input.try_dev_data_ptr()?;
        let input_dev_data = input_dev_data_ref.data::<I>();

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let output = ctx.get_output(0)?;
        let mut output_dev_data_ref = output.dev_data_ptr_mut();
        let mut output_dev_data = output_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        let rank = output.shape().len();

        let info_buffer = ctx.execution_state().scratch_alloc().allocate(2 * rank)?;
        info_buffer[..rank].copy_from_slice(output.shape());
        info_buffer[rank..2 * rank].copy_from_slice(input.stride());

        unsafe {
            rmlk_cuda::kernels::unary::explicit_io_types_compute::<I, O>(
                &self.stream,
                kernel,
                &input_dev_data,
                &mut output_dev_data,
            )?;
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let in_dtype = ctx.get_input(0)?.dtype();
        let out_dtype = ctx
            .get_attributes()
            .map(|attrs| cast::get_value(&attrs))
            .transpose()
            .map(Option::flatten)?
            .ok_or_else(|| InternalError::MissingAttribute {
                name: "missing `to` cast attribute".to_string(),
            })?;

        match (in_dtype, out_dtype) {
            (DataType::Float16, DataType::Float) => self.compute_cast::<f16, f32>(ctx),
            (DataType::Float16, DataType::Double) => self.compute_cast::<f16, f64>(ctx),
            (DataType::Float16, DataType::Int32) => self.compute_cast::<f16, i32>(ctx),
            (DataType::Float16, DataType::Uint32) => self.compute_cast::<f16, u32>(ctx),
            (DataType::Float16, DataType::Int64) => self.compute_cast::<f16, i64>(ctx),
            (DataType::Float16, DataType::Uint64) => self.compute_cast::<f16, u64>(ctx),
            (DataType::Float, DataType::Float16) => self.compute_cast::<f32, f16>(ctx),
            (DataType::Float, DataType::Double) => self.compute_cast::<f32, f64>(ctx),
            (DataType::Float, DataType::Int32) => self.compute_cast::<f32, i32>(ctx),
            (DataType::Float, DataType::Uint32) => self.compute_cast::<f32, u32>(ctx),
            (DataType::Float, DataType::Int64) => self.compute_cast::<f32, i64>(ctx),
            (DataType::Float, DataType::Uint64) => self.compute_cast::<f32, u64>(ctx),
            (DataType::Double, DataType::Float16) => self.compute_cast::<f64, f16>(ctx),
            (DataType::Double, DataType::Float) => self.compute_cast::<f64, f32>(ctx),
            (DataType::Double, DataType::Int32) => self.compute_cast::<f64, i32>(ctx),
            (DataType::Double, DataType::Uint32) => self.compute_cast::<f64, u32>(ctx),
            (DataType::Double, DataType::Int64) => self.compute_cast::<f64, i64>(ctx),
            (DataType::Double, DataType::Uint64) => self.compute_cast::<f64, u64>(ctx),
            (DataType::Int32, DataType::Float16) => self.compute_cast::<i32, f16>(ctx),
            (DataType::Int32, DataType::Float) => self.compute_cast::<i32, f32>(ctx),
            (DataType::Int32, DataType::Double) => self.compute_cast::<i32, f64>(ctx),
            (DataType::Int32, DataType::Int64) => self.compute_cast::<i32, i64>(ctx),
            (DataType::Uint32, DataType::Float16) => self.compute_cast::<u32, f16>(ctx),
            (DataType::Uint32, DataType::Float) => self.compute_cast::<u32, f32>(ctx),
            (DataType::Uint32, DataType::Double) => self.compute_cast::<u32, f64>(ctx),
            (DataType::Uint32, DataType::Uint64) => self.compute_cast::<u32, u64>(ctx),
            (DataType::Int64, DataType::Float16) => self.compute_cast::<i64, f16>(ctx),
            (DataType::Int64, DataType::Float) => self.compute_cast::<i64, f32>(ctx),
            (DataType::Int64, DataType::Double) => self.compute_cast::<i64, f64>(ctx),
            (DataType::Int64, DataType::Int32) => self.compute_cast::<i64, i32>(ctx),
            (DataType::Uint64, DataType::Float16) => self.compute_cast::<u64, f16>(ctx),
            (DataType::Uint64, DataType::Float) => self.compute_cast::<u64, f32>(ctx),
            (DataType::Uint64, DataType::Double) => self.compute_cast::<u64, f64>(ctx),
            (DataType::Uint64, DataType::Uint32) => self.compute_cast::<u64, u32>(ctx),
            _ => Err(CastError::UnsupportedCast {
                src: in_dtype,
                dst: out_dtype,
            }
            .into()),
        }
    }
}

#[derive(Debug)]
pub enum CastError {
    UnsupportedCast { src: DataType, dst: DataType },
}

impl Display for CastError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CastError::UnsupportedCast { src, dst } => {
                write!(f, "unsupported cast {:?} to {:?}", src, dst)
            }
        }
    }
}

impl std::error::Error for CastError {}
