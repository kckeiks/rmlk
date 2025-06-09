use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::{attributes, utils};
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::{Num, ToPrimitive};
use rmlk_schema::{DataType, DataTypeMap, Op};
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

    pub fn compute_shape<D, T>(mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: ShapeProcessor,
    {
        let data = ctx.get_input(0)?;
        let rank = data.shape().len();

        let raw_start = ctx
            .get_attributes()
            .map(|attrs| attributes::shape::get_start(&attrs))
            .unwrap_or(0);
        let raw_end = match ctx
            .get_attributes()
            .and_then(|attrs| attributes::shape::get_end(&attrs))
        {
            None => rank.to_i32().ok_or(InternalError::UnsupportedRankSize {
                message: format!("failed to convert `{rank}` to i32"),
            })?,
            Some(end) => end,
        };

        let (start, end) = utils::derive_range(raw_start as i64, raw_end as i64, rank)?;

        self.compute_output_shape(start, end, ctx)?;

        let output = ctx.get_output(0)?;
        common::init_tensor_device_data::<i64>(&self.stream, output)?;

        let shape = ctx.get_output(0)?;
        let mut shape_ptr = shape.try_dev_data_ptr_mut()?;
        let mut shape_dev_data = shape_ptr.data_mut::<i64>();

        let data = ctx.get_input(0)?;
        let shape_host_buf = ctx
            .execution_state()
            .scratch_alloc()
            .allocate_and_convert_from_slice::<_, i64>(data.shape())?;

        T::compute::<i64>(
            &self.stream,
            &shape_host_buf[start..end],
            &mut shape_dev_data,
        )
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: ShapeProcessor,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_shape::<f32, T>(ctx),
            DataType::Int64 => self.compute_shape::<i64, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Shape,
                dtype,
            }),
        }
    }
}

pub trait ShapeProcessor {
    fn compute<D>(
        stream: &Arc<CudaStream>,
        shape: &[D],
        output_dev_data: &mut CudaSlice<D>,
    ) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct DefaultShapeProcessor(());

impl ShapeProcessor for DefaultShapeProcessor {
    fn compute<D>(
        stream: &Arc<CudaStream>,
        shape: &[D],
        output_dev_data: &mut CudaSlice<D>,
    ) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        stream
            .memcpy_htod(shape, output_dev_data)
            .map_err(Into::into)
    }
}

pub struct NoOpShapeProcessor(());

impl ShapeProcessor for NoOpShapeProcessor {
    fn compute<D>(_: &Arc<CudaStream>, _: &[D], _: &mut CudaSlice<D>) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        Ok(())
    }
}
