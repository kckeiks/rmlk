use crate::attributes;
use crate::core::error::InternalError;
use crate::core::Context;
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
    _stream: Arc<CudaStream>,
}

impl ConstantBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            _stream: stream.clone(),
        }
    }

    fn load_from_values<T>(
        &mut self,
        values: &[T],
        ctx: &Context<Cuda>,
        is_scalar: bool,
    ) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromBytes,
    {
        let output_tensor = ctx.get_output(0)?;
        if is_scalar {
            output_tensor.copy_shape_from_slice(&[]);
            output_tensor.init_scalar_payload::<T>()?;
        } else {
            output_tensor.copy_shape_from_slice(&[values.len()]);
            output_tensor.init_payload::<T>()?;
        }

        debug!(
            "[output][dtype={:?}][shape={:?}][strides={:?}]",
            output_tensor.dtype(),
            output_tensor.shape(),
            output_tensor.stride()
        );

        output_tensor.write_payload_from_slice(values)?;

        Ok(())
    }

    fn load_from_bytes<T>(&self, shape: &[usize], bytes: &[u8], ctx: &Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromBytes + Unpin,
    {
        let output_tensor = ctx.get_output(0)?;
        output_tensor.copy_shape_from_slice(shape);

        let output_tensor = ctx.get_output(0)?;
        output_tensor.init_payload::<T>()?;

        debug!(
            "[output][dtype={:?}][shape={:?}][strides={:?}]",
            output_tensor.dtype(),
            output_tensor.shape(),
            output_tensor.stride()
        );

        let data = T::from_bytes(bytes)?;
        output_tensor.write_payload_from_slice::<T>(&data)?;

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
            return self.load_from_values::<f32>(&[value], ctx, true);
        }

        if let Some(value) = attributes::constant::get_floats(&attrs) {
            return self.load_from_values::<f32>(value, ctx, false);
        }

        if let Some(value) = attributes::constant::get_int(&attrs) {
            return self.load_from_values::<i32>(&[value], ctx, true);
        }

        if let Some(value) = attributes::constant::get_ints(&attrs) {
            return self.load_from_values::<i32>(value, ctx, false);
        }

        Err(InternalError::MissingAttributes.into())
    }
}
