use crate::core::error::InternalError;
use crate::core::Context;

#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::{Float, Num, NumCast, ToPrimitive};
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::Debug;
use std::sync::Arc;

pub struct RangeBackend {
    _stream: Arc<CudaStream>,
}

impl RangeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            _stream: stream.clone(),
        }
    }

    fn compute_range<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: Copy
            + DataTypeMap
            + CudnnDataType
            + Default
            + DeviceRepr
            + Num
            + NumCast
            + ValidAsZeroBits
            + ElementCount
            + Debug,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let start_value = scratch_alloc.allocate::<T>(1)?;
        let limit_value = scratch_alloc.allocate::<T>(1)?;
        let delta_value = scratch_alloc.allocate::<T>(1)?;

        let start_tensor = ctx.get_input(0)?;

        debug!(
            "[start][dtype={:?}][shape={:?}][stride=[{:?}]",
            start_tensor.dtype(),
            start_tensor.shape(),
            start_tensor.stride(),
        );

        start_tensor.payload_to_host(start_value)?;

        let limit_tensor = ctx.get_input(1)?;

        debug!(
            "[limit][dtype={:?}][shape={:?}][stride=[{:?}]",
            limit_tensor.dtype(),
            limit_tensor.shape(),
            limit_tensor.stride(),
        );

        limit_tensor.payload_to_host(limit_value)?;

        let delta_tensor = ctx.get_input(2)?;

        debug!(
            "[delta][dtype={:?}][shape={:?}][stride=[{:?}]",
            delta_tensor.dtype(),
            delta_tensor.shape(),
            delta_tensor.stride(),
        );

        delta_tensor.payload_to_host(delta_value)?;

        let start = start_value[0];
        let limit = limit_value[0];
        let delta = delta_value[0];

        debug!("[start={:?}][limit={:?}][delta=[{:?}]", start, limit, delta);

        // First compute N = the number of elements.
        let elem_count = T::element_count(start, limit, delta)?;

        // The shape of the output should be [N].
        let output_shape = scratch_alloc.allocate_fill(1, elem_count)?;

        let output_tensor = ctx.get_output(0)?;
        output_tensor.copy_shape_from_slice(output_shape);
        output_tensor.init_payload::<T>()?;

        debug!(
            "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
            output_tensor.dtype(),
            output_tensor.shape(),
            output_tensor.stride(),
        );

        let output = scratch_alloc.allocate_fill::<T>(elem_count, T::zero())?;

        // Compute the values in the output on the host.
        compute_output(start, delta, elem_count, output)?;

        // Copy the data from the host to the device.
        output_tensor.write_payload_from_slice(output)?;

        #[cfg(feature = "debugger")]
        debug::write_results_ternary::<T, T, T, T>(
            "debugging/range",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_range::<f16>(ctx),
            DataType::Float => self.compute_range::<f32>(ctx),
            DataType::Double => self.compute_range::<f64>(ctx),
            DataType::Int32 => self.compute_range::<i32>(ctx),
            DataType::Int64 => self.compute_range::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

pub trait ElementCount {
    fn element_count(start: Self, limit: Self, delta: Self) -> Result<usize>;
}

impl ElementCount for i32 {
    fn element_count(start: i32, limit: i32, delta: i32) -> Result<usize> {
        let count = ((limit - start) / delta).max(0);
        usize::try_from(count).map_err(|_| InternalError::UnableToConvertValue.into())
    }
}

impl ElementCount for i64 {
    fn element_count(start: i64, limit: i64, delta: i64) -> Result<usize> {
        let count = ((limit - start) / delta).max(0);
        usize::try_from(count).map_err(|_| InternalError::UnableToConvertValue.into())
    }
}

impl ElementCount for f16 {
    fn element_count(start: f16, limit: f16, delta: f16) -> Result<usize> {
        let count = ((limit - start) / delta).ceil();
        if count < f16::from_f32(0.0) {
            return Err(InternalError::UnableToConvertValue.into());
        }
        // Todo: circle back about this.
        if count > f16::MAX {
            return Err(InternalError::UnableToConvertValue.into());
        }
        count
            .to_usize()
            .ok_or(InternalError::UnableToConvertValue.into())
    }
}

impl ElementCount for f32 {
    fn element_count(start: f32, limit: f32, delta: f32) -> Result<usize> {
        let count = ((limit - start) / delta).ceil();
        if count < 0.0 {
            return Err(InternalError::UnableToConvertValue.into());
        }
        if count > (usize::MAX as f32) {
            return Err(InternalError::UnableToConvertValue.into());
        }
        Ok(count as usize)
    }
}

impl ElementCount for f64 {
    fn element_count(start: f64, limit: f64, delta: f64) -> Result<usize> {
        let count = ((limit - start) / delta).ceil();
        if count < 0.0 {
            return Err(InternalError::UnableToConvertValue.into());
        }
        if count > (usize::MAX as f64) {
            return Err(InternalError::UnableToConvertValue.into());
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
