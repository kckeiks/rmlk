use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, DeviceRepr, ValidAsZeroBits};
use num_traits::{Num, NumCast};
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct RangeBackend {
    device: Arc<CudaDevice>,
}

impl RangeBackend {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
        }
    }

    fn compute_range<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: Copy + DataTypeMap + CudnnDataType + Default + DeviceRepr + Num + NumCast + ValidAsZeroBits + ElementCount,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let start_value = scratch_alloc.allocate::<I>(1)?;
        let limit_value = scratch_alloc.allocate::<I>(1)?;
        let delta_value = scratch_alloc.allocate::<I>(1)?;

        {
            let start_tensor = ctx.get_input(0)?;
            let start_ptr = start_tensor.try_dev_data_ptr()?;
            let start_view = start_ptr.data::<I>();
            self.device.dtoh_sync_copy_into(start_view.as_ref(), start_value)?;

            let limit_tensor = ctx.get_input(1)?;
            let limit_ptr = limit_tensor.try_dev_data_ptr()?;
            let limit_view = limit_ptr.data::<I>();
            self.device.dtoh_sync_copy_into(limit_view.as_ref(), limit_value)?;


            let delta_tensor = ctx.get_input(2)?;
            let delta_ptr = delta_tensor.try_dev_data_ptr()?;
            let delta_view = delta_ptr.data::<I>();
            self.device.dtoh_sync_copy_into(delta_view.as_ref(), delta_value)?;
        }

        let start = start_value[0];
        let limit = limit_value[0];
        let delta = delta_value[0];

        // First compute N = the number of elements.
        let elem_count = I::element_count(start, limit, delta)?;

        // The shape of the output should be [N].
        let output_shape = scratch_alloc.allocate_fill(1, elem_count)?;

        let output_tensor = ctx.get_output(0)?;
        let dst_id = output_tensor.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst_id)?;

        // Try to init the tensor.
        let output_tensor = ctx.get_output(0)?;
        common::init_tensor_device_data::<I>(&self.device, output_tensor)?;

        let output = scratch_alloc.allocate_fill::<I>(elem_count, I::zero())?;

        // Compute the values in the output on the host.
        compute_output(start, delta, elem_count, output)?;

        // Copy the data from the host to the device.
        let output_tensor = ctx.get_output(0)?;
        let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_ptr.data_mut::<I>();
        self.device.htod_sync_copy_into(&output, output_view.as_mut())?;

        #[cfg(debug_assertions)]
        {
            use log::debug;

            let start = ctx.get_input(0)?;
            let limit = ctx.get_input(1)?;
            let delta = ctx.get_input(2)?;
            let output = ctx.get_output(0)?;
            debug!(
                "[start][shape={:?}][stride=[stride=[{:?}]",
                start.shape(),
                start.stride(),
            );
            debug!(
                "[limit][shape={:?}][stride=[stride=[{:?}]",
                limit.shape(),
                limit.stride(),
            );
            debug!(
                "[delta][shape={:?}][stride=[stride=[{:?}]",
                delta.shape(),
                delta.stride(),
            );
            debug!(
                "[output][shape={:?}][stride=[stride=[{:?}]",
                output.shape(),
                output.stride()
            );
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_range::<f32>(ctx),
            DataType::Int32 => self.compute_range::<i32>(ctx),
            DataType::Int64 => self.compute_range::<i64>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Unsqueeze,
                dtype,
            }),
        }
    }
}

pub trait ElementCount {
    fn element_count(start: Self, limit: Self, delta: Self) -> Result<usize>;
}

impl ElementCount for i32 {
    fn element_count(start: i32, limit: i32, delta: i32) -> Result<usize> {
        let count = ((limit - start) / delta).max(0);
        usize::try_from(count).map_err(|_| InternalError::UnableToConvertValue)
    }
}

impl ElementCount for i64 {
    fn element_count(start: i64, limit: i64, delta: i64) -> Result<usize> {
        let count = ((limit - start) / delta).max(0);
        usize::try_from(count).map_err(|_| InternalError::UnableToConvertValue)
    }
}

impl ElementCount for f32 {
    fn element_count(start: f32, limit: f32, delta: f32) -> Result<usize> {
        let count = ((limit - start) / delta).ceil();
        if count < 0.0 {
            return Err(InternalError::UnableToConvertValue);
        }
        if count > (usize::MAX as f32) {
            return Err(InternalError::UnableToConvertValue);
        }
        Ok(count as usize)
    }
}

#[inline]
fn compute_output<T>(start: T, delta: T, elem_count: usize, output: &mut [T]) -> Result<()>
where
    T: Copy + Num + NumCast,
{
    for idx in 0..elem_count {
        let i = T::from(idx).ok_or(InternalError::UnableToConvertValue)?;
        output[idx] = start + (i * delta);
    }

    Ok(())
}