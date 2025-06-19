use crate::attributes;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils::FromBytes;
use anyhow::Result;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct ConstantBackend {
    stream: Arc<CudaStream>,
}

impl ConstantBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_from_values<T>(&mut self, values: &[T], ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromBytes,
    {
        {
            let output_tensor = ctx.get_output(0)?;
            let dst_id = output_tensor.dst_id();
            ctx.execution_state_mut()
                .copy_shape_from_slice(&[values.len()], dst_id)?;
        }

        let output_tensor = ctx.get_output(0)?;

        debug!(
            "[output][shape={:?}][strides={:?}]",
            output_tensor.shape(),
            output_tensor.stride()
        );

        common::init_tensor_device_data::<T>(&self.stream, output_tensor)?;

        let output_tensor = ctx.get_output(0)?;
        let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_ptr.data_mut::<T>();
        self.stream.memcpy_htod(values, output_view.as_mut())?;

        Ok(())
    }

    fn load_from_bytes<T>(
        &self,
        shape: &[usize],
        bytes: &[u8],
        ctx: &mut Context<Cuda>,
    ) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromBytes + Unpin,
    {
        {
            let output_tensor = ctx.get_output(0)?;
            let dst_id = output_tensor.dst_id();
            ctx.execution_state_mut()
                .copy_shape_from_slice(shape, dst_id)?;
        }

        let output_tensor = ctx.get_output(0)?;

        debug!(
            "[output][shape={:?}][strides={:?}]",
            output_tensor.shape(),
            output_tensor.stride()
        );

        common::init_tensor_device_data::<T>(&self.stream, output_tensor)?;

        let data = T::from_bytes(bytes)?;
        let output_tensor = ctx.get_output(0)?;
        let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_ptr.data_mut::<T>();
        self.stream.memcpy_htod(&data, output_view.as_mut())?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)
            .map_err(Box::new)?;

        debug!("[attributes={:?}]", attrs);

        if let Some((dtype, shape, bytes)) = attributes::constant::get_raw_value(&attrs) {
            return match dtype {
                DataType::Float16 => self.load_from_bytes::<f16>(
                    shape,
                    bytes.ok_or(InternalError::MissingAttributes)?,
                    ctx,
                ),
                DataType::Float => self.load_from_bytes::<f32>(
                    shape,
                    bytes.ok_or(InternalError::MissingAttributes)?,
                    ctx,
                ),
                DataType::Double => self.load_from_bytes::<f64>(
                    shape,
                    bytes.ok_or(InternalError::MissingAttributes)?,
                    ctx,
                ),
                DataType::Int32 => self.load_from_bytes::<i32>(
                    shape,
                    bytes.ok_or(InternalError::MissingAttributes)?,
                    ctx,
                ),
                DataType::Uint32 => self.load_from_bytes::<u32>(
                    shape,
                    bytes.ok_or(InternalError::MissingAttributes)?,
                    ctx,
                ),
                DataType::Int64 => self.load_from_bytes::<i64>(
                    shape,
                    bytes.ok_or(InternalError::MissingAttributes)?,
                    ctx,
                ),
                DataType::Uint64 => self.load_from_bytes::<u64>(
                    shape,
                    bytes.ok_or(InternalError::MissingAttributes)?,
                    ctx,
                ),
                _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
            };
        }

        if let Some(value) = attributes::constant::get_float(&attrs) {
            return self.load_from_values::<f32>(&[value], ctx);
        }

        if let Some(value) = attributes::constant::get_floats(&attrs) {
            return self.load_from_values::<f32>(value, ctx);
        }

        if let Some(value) = attributes::constant::get_int(&attrs) {
            return self.load_from_values::<i32>(&[value], ctx);
        }

        if let Some(value) = attributes::constant::get_ints(&attrs) {
            return self.load_from_values::<i32>(value, ctx);
        }

        Err(InternalError::MissingAttributes.into())
    }
}
