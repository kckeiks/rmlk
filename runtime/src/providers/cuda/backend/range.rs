use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::{Num, NumCast};
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct RangeBackend {
    stream: Arc<CudaStream>,
}

impl RangeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn compute_range<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: Copy
            + DataTypeMap
            + CudnnDataType
            + Default
            + DeviceRepr
            + Num
            + NumCast
            + ValidAsZeroBits
            + ElementCount,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let start_value = scratch_alloc.allocate::<I>(1)?;
        let limit_value = scratch_alloc.allocate::<I>(1)?;
        let delta_value = scratch_alloc.allocate::<I>(1)?;

        {
            let start_tensor = ctx.get_input(0)?;
            let start_ptr = start_tensor.try_dev_data_ptr()?;
            let start_view = start_ptr.data::<I>();
            self.stream.memcpy_dtoh(start_view.as_ref(), start_value)?;

            let limit_tensor = ctx.get_input(1)?;
            let limit_ptr = limit_tensor.try_dev_data_ptr()?;
            let limit_view = limit_ptr.data::<I>();
            self.stream.memcpy_dtoh(limit_view.as_ref(), limit_value)?;

            let delta_tensor = ctx.get_input(2)?;
            let delta_ptr = delta_tensor.try_dev_data_ptr()?;
            let delta_view = delta_ptr.data::<I>();
            self.stream.memcpy_dtoh(delta_view.as_ref(), delta_value)?;
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
        common::init_tensor_device_data::<I>(&self.stream, output_tensor)?;

        let output = scratch_alloc.allocate_fill::<I>(elem_count, I::zero())?;

        // Compute the values in the output on the host.
        compute_output(start, delta, elem_count, output)?;

        // Copy the data from the host to the device.
        let output_tensor = ctx.get_output(0)?;
        let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_ptr.data_mut::<I>();
        self.stream.memcpy_htod(output, output_view.as_mut())?;

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
                op: Op::Range,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_positive_int_step() {
        let count = i32::element_count(3, 9, 3).unwrap();
        assert_eq!(count, 2);
        let mut output = vec![0; count];
        compute_output(3, 3, count, &mut output).unwrap();
        assert_eq!(output, [3, 6]);
    }

    #[test]
    fn test_basic_negative_int_step() {
        let count = i32::element_count(10, 4, -2).unwrap();
        assert_eq!(count, 3);
        let mut output = vec![0; count];
        compute_output(10, -2, count, &mut output).unwrap();
        assert_eq!(output, [10, 8, 6]);
    }

    #[test]
    fn test_positive_float_step() {
        let count = f32::element_count(0.1, 1.0, 0.3).unwrap();
        assert_eq!(count, 3);
        let mut output = vec![0.0; count];
        compute_output(0.1, 0.3, count, &mut output).unwrap();
        assert_eq!(output, [0.1, 0.4, 0.7]);
    }

    #[test]
    fn test_negative_float_step() {
        let count = f32::element_count(2.0, 1.0, -0.3).unwrap();
        assert_eq!(count, 4);
        let mut output = vec![0.0; count];
        compute_output(2.0, -0.3, count, &mut output).unwrap();
        assert_eq!(output, [2.0, 1.7, 1.4, 1.1]);
    }

    #[test]
    fn test_start_equal_limit() {
        let count = i32::element_count(5, 5, 1).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_incompatible_delta_positive() {
        let count = i32::element_count(5, 0, 1).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_incompatible_delta_negative() {
        let count = i32::element_count(0, 5, -1).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_zero_delta_handling() {
        let result = f32::element_count(1.0, 2.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_range() {
        let count = i64::element_count(0, 1_000_000, 1).unwrap();
        assert_eq!(count, 1_000_000);
    }

    #[test]
    fn test_large_negative_range() {
        let count = i64::element_count(1_000_000, 0, -1).unwrap();
        assert_eq!(count, 1_000_000);
    }

    #[test]
    fn test_float_precision_edge() {
        let count = f32::element_count(0.0, 1.0, 0.333).unwrap();
        assert_eq!(count, 4);
        let mut output = vec![0.0; count];
        compute_output(0.0, 0.333, count, &mut output).unwrap();
        assert!((output[3] < 1.0));
    }
}
