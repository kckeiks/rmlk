use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
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
    stream: Arc<CudaStream>,
}

impl ShapeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    pub fn compute_output_shape(
        &mut self,
        start: usize,
        end: usize,
        ctx: &mut Context<Cuda>,
    ) -> Result<()> {
        let alloc = ctx.execution_state().scratch_alloc().clone();
        let tensor_shape_buf = alloc.allocate_fill::<usize>(1, end - start)?;

        let tensor = ctx.get_output(0)?;
        let dst_id = tensor.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(tensor_shape_buf, dst_id)?;

        Ok(())
    }

    pub fn compute_shape<D>(mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
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

            self.compute_output_shape(start, end, ctx)?;

            let output = ctx.get_output(0)?;

            debug!(
                "[output][dtype={:?}][shape={:?}][strides={:?}]",
                output.dtype(),
                output.shape(),
                output.stride()
            );

            common::init_tensor_device_data::<i64>(&self.stream, output)?;

            let shape = ctx.get_output(0)?;
            let mut shape_ptr = shape.try_dev_data_ptr_mut()?;
            let mut shape_dev_data = shape_ptr.data_mut::<i64>();

            let data = ctx.get_input(0)?;
            let shape_host_buf = ctx
                .execution_state()
                .scratch_alloc()
                .allocate_and_convert_from_slice::<_, i64>(data.shape())?;

            self.stream
                .memcpy_htod(&shape_host_buf[start..end], shape_dev_data.as_mut())
                .map_err(|e| InternalError::Device { error: e.into() })?;
        }

        common::write_results_shape::<D>("debugging/shape", self.stream.clone(), ctx)?;

        /*self.stream
            .synchronize()
            .map_err(|e| InternalError::Device { error: e.into() })?;*/

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
