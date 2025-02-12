use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::{attributes, utils};
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtrMut, DeviceRepr, ValidAsZeroBits};
use num_traits::{Num, ToPrimitive};
use rmlk_schema::{Attribute, DataType, DataTypeMap, Op};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

pub struct ShapeBackend {
    device: Arc<CudaDevice>,
}

impl ShapeBackend {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn compute_output_shape(&mut self, size: usize, ctx: &mut Context<Cuda>) -> Result<()> {
        let alloc = ctx.execution_state().scratch_alloc().clone();

        let output_shape_buf = alloc.allocate::<usize>(size)?;

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape_buf, dst_id)?;

        Ok(())
    }

    pub fn compute_shape<D, T>(mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: ShapeProcessor,
    {
        let data = ctx.get_input(0)?;
        let rank = data.shape().len();

        let start = ctx
            .get_attributes()
            .map(attributes::shape::get_start)
            .unwrap_or(0);
        let end = match ctx.get_attributes().map(attributes::shape::get_start) {
            None => rank.to_i32().ok_or(InternalError::UnsupportedRankSize {
                message: format!("failed to convert `{rank}` to i32"),
            })?,
            Some(end) => end,
        };

        let (start, end) = utils::derive_range(start as i64, end as i64, rank)?;

        self.compute_output_shape(end.checked_sub(start).unwrap_or(0), ctx)?;

        let output = ctx.get_output(0)?;
        common::init_tensor_device_data::<i64>(&self.device, output)?;

        let output = ctx.get_output(0)?;
        let mut output_ptr = output.try_dev_data_ptr_mut()?;

        let mut dev_data = output_ptr.data_mut::<i64>();

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let output_shape_buf = alloc.allocate_and_convert_from_slice::<_, i64>(output.shape())?;

        T::compute::<i64>(&self.device, &output_shape_buf[start..end], &mut dev_data)
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: ShapeProcessor,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_shape::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::NoOp,
                dtype,
            }),
        }
    }
}

pub trait ShapeProcessor {
    fn compute<D>(
        device: &Arc<CudaDevice>,
        shape: &[D],
        output_dev_data: &mut CudaSlice<D>,
    ) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct DefaultShapeProcessor(());

impl ShapeProcessor for DefaultShapeProcessor {
    fn compute<D>(
        device: &Arc<CudaDevice>,
        shape: &[D],
        output_dev_data: &mut CudaSlice<D>,
    ) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        device
            .htod_sync_copy_into(shape, output_dev_data)
            .map_err(Into::into)
    }
}

pub struct NoOpShapeProcessor(());

impl ShapeProcessor for NoOpShapeProcessor {
    fn compute<D>(_: &Arc<CudaDevice>, _: &[D], _: &mut CudaSlice<D>) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        Ok(())
    }
}
