use crate::attributes;
use crate::attributes::constant_of_shape::AttributeTensor;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_schema::DataTypeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct ConstantOfShapeBackend {
    stream: Arc<CudaStream>,
}

impl ConstantOfShapeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn compute_constant_of_shape<O>(&mut self, ctx: &mut Context<Cuda>, value: O) -> Result<()>
    where
        O: Copy + Default + DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let input_data = {
            let input = ctx.get_input(0)?;

            debug!(
                "[input][dtype=i64][shape={:?}][strides={:?}]",
                input.shape(),
                input.stride()
            );

            let input_size = input.shape().iter().product();
            let on_host_data = scratch_alloc.allocate::<i64>(input_size)?;

            let data_ptr = input.try_dev_data_ptr()?;
            let dev_view = data_ptr.data::<i64>();

            self.stream
                .memcpy_dtoh(dev_view.as_ref(), on_host_data)
                .map_err(|e| InternalError::Device { error: e.into() })?;

            on_host_data
        };

        let output_shape = scratch_alloc.allocate_and_convert_from_slice(input_data)?;
        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst_id)?;

        let output = ctx.get_output(0)?;
        common::init_tensor_device_data::<O>(&self.stream, output)?;

        let output = ctx.get_output(0)?;

        debug!(
            "[output][dtype={:?}][shape={:?}][strides={:?}]",
            output.dtype(),
            output.shape(),
            output.stride()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let on_host_data =
            scratch_alloc.allocate_fill::<O>(output.shape().iter().product(), value)?;

        let mut output_data_ptr = output.try_dev_data_ptr_mut()?;
        let mut output_data_view = output_data_ptr.data_mut::<O>();

        self.stream
            .memcpy_htod(on_host_data, output_data_view.as_mut())
            .map_err(|e| InternalError::Device { error: e.into() })?;

        /*self.stream
        .synchronize()
        .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let attrs = ctx.get_attributes().clone();

        if let Some(attrs) = ctx.get_attributes() {
            debug!("[attributes={:?}]", attrs);
        }

        match attrs
            .as_ref()
            .map(|attrs| attributes::constant_of_shape::get_value(attrs))
            .transpose()?
            .flatten()
        {
            Some(AttributeTensor::F16(data)) => self.compute_constant_of_shape::<f16>(
                ctx,
                *data
                    .first()
                    .ok_or(Box::new(ConstOfShapeError::EmptyAttributeTensor))?,
            ),
            Some(AttributeTensor::F32(data)) => self.compute_constant_of_shape::<f32>(
                ctx,
                *data
                    .first()
                    .ok_or(Box::new(ConstOfShapeError::EmptyAttributeTensor))?,
            ),
            Some(AttributeTensor::F64(data)) => self.compute_constant_of_shape::<f64>(
                ctx,
                *data
                    .first()
                    .ok_or(Box::new(ConstOfShapeError::EmptyAttributeTensor))?,
            ),
            Some(AttributeTensor::I32(data)) => self.compute_constant_of_shape::<i32>(
                ctx,
                *data
                    .first()
                    .ok_or(Box::new(ConstOfShapeError::EmptyAttributeTensor))?,
            ),
            Some(AttributeTensor::I64(data)) => self.compute_constant_of_shape::<i64>(
                ctx,
                *data
                    .first()
                    .ok_or(Box::new(ConstOfShapeError::EmptyAttributeTensor))?,
            ),
            None => self.compute_constant_of_shape::<f32>(ctx, 0.0),
        }
    }
}

#[derive(Debug)]
pub enum ConstOfShapeError {
    EmptyAttributeTensor,
}

impl Display for ConstOfShapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ConstOfShapeError {}
