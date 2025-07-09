use crate::core::error::InternalError;
use crate::core::Context;

#[cfg(feature = "dump")]
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::{attributes, utils};
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::{Num, ToPrimitive};
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct ShapeBackend {
    _stream: Arc<CudaStream>,
}

impl ShapeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            _stream: stream.clone(),
        }
    }

    pub fn compute_shape<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let data = ctx.get_input(0)?;

        debug!(
            "[data][dtype={:?}][shape={:?}][strides={:?}]",
            data.dtype(),
            data.shape(),
            data.stride()
        );

        let rank = data.shape().len();

        let attrs = ctx.get_attributes();

        debug!("[attributes={attrs:?}]");

        let raw_start = attrs
            .as_ref()
            .map(|attrs| attributes::shape::get_start(attrs.as_ref()))
            .unwrap_or(0);
        let raw_end = match attrs.and_then(|attrs| attributes::shape::get_end(&attrs)) {
            None => rank.to_i32().ok_or(InternalError::UnsupportedRankSize {
                message: format!("failed to convert `{rank}` to i32"),
            })?,
            Some(end) => end,
        };

        let (start, end) = utils::derive_range(raw_start as i64, raw_end as i64, rank)?;

        compute_output_shape(start, end, ctx)?;

        let output = ctx.get_output(0)?;
        output.init_payload::<i64>()?;

        debug!(
            "[output][dtype={:?}][shape={:?}][strides={:?}]",
            output.dtype(),
            output.shape(),
            output.stride()
        );

        let shape = ctx.get_output(0)?;

        let shape_host_buf = ctx
            .execution_state()
            .scratch_alloc()
            .allocate_and_convert_from_slice::<_, i64>(&data.shape())?;

        shape.write_payload_from_slice(&shape_host_buf[start..end])?;

        #[cfg(feature = "dump")]
        debug::write_results_shape::<T>("debugging/shape", self.stream.clone(), ctx)?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_shape::<f16>(ctx),
            DataType::Float => self.compute_shape::<f32>(ctx),
            DataType::Double => self.compute_shape::<f64>(ctx),
            DataType::Int32 => self.compute_shape::<i32>(ctx),
            DataType::Int64 => self.compute_shape::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

pub fn compute_output_shape(start: usize, end: usize, ctx: &Context<Cuda>) -> Result<()> {
    let alloc = ctx.execution_state().scratch_alloc().clone();
    let tensor_shape_buf = alloc.allocate_fill::<usize>(1, end - start)?;
    let tensor = ctx.get_output(0)?;
    tensor.copy_shape_from_slice(tensor_shape_buf);
    Ok(())
}
