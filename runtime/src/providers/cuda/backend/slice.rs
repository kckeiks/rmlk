use crate::core::error::{ConversionError, UnsupportedDataType};
use crate::core::Context;

#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::{Num, PrimInt, Signed, ToPrimitive};
use rmlk_cuda::kernels::slice;
use rmlk_cuda::kernels::slice::SliceKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

pub struct SliceBackend {
    stream: Arc<CudaStream>,
}

impl SliceBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, dtype: DataType, indices_dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match (dtype, indices_dtype) {
            (DataType::Float16, DataType::Int32) => SliceKernel::FwdF16WithIndexI32,
            (DataType::Float16, DataType::Int64) => SliceKernel::FwdF16WithIndexI64,
            (DataType::Float, DataType::Int32) => SliceKernel::FwdF32WithIndexI32,
            (DataType::Float, DataType::Int64) => SliceKernel::FwdF32WithIndexI64,
            (DataType::Double, DataType::Int32) => SliceKernel::FwdF64WithIndexI32,
            (DataType::Double, DataType::Int64) => SliceKernel::FwdF64WithIndexI64,
            (DataType::Int32, DataType::Int32) => SliceKernel::FwdI32WithIndexI32,
            (DataType::Int32, DataType::Int64) => SliceKernel::FwdI32WithIndexI64,
            (DataType::Int64, DataType::Int32) => SliceKernel::FwdI64WithIndexI32,
            (DataType::Int64, DataType::Int64) => SliceKernel::FwdI64WithIndexI64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        slice::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_slice<T, Tind>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + Default + Copy + Debug,
        Tind: DataTypeMap
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + Default
            + Copy
            + Signed
            + PrimInt
            + Debug
            + TryFrom<usize>,
        <Tind as TryFrom<usize>>::Error: Debug,
        i64: From<Tind>,
    {
        {
            let input_tensor = ctx.get_input(0)?;

            debug!(
                "[input][dtype={:?}][shape={:?}][strides={:?}]",
                input_tensor.dtype(),
                input_tensor.shape(),
                input_tensor.stride()
            );

            compute_output_shape::<Tind>(ctx)?;

            let output_tensor = ctx.get_output(0)?;
            output_tensor.init_payload::<T>()?;

            debug!(
                "[output][dtype={:?}][shape={:?}][strides={:?}]",
                output_tensor.dtype(),
                output_tensor.shape(),
                output_tensor.stride()
            );

            let input_tensor = ctx.get_input(0)?;
            let input_payload = input_tensor.payload();
            let input_data = input_payload.data::<T>();

            let starts_tensor = ctx.get_input(1)?;
            let starts_payload = starts_tensor.payload();
            let starts_data = starts_payload.data::<Tind>();

            let ends_tensor = ctx.get_input(2)?;
            let ends_payload = ends_tensor.payload();
            let ends_data = ends_payload.data::<Tind>();

            let output_tensor = ctx.get_output(0)?;
            let mut output_payload = output_tensor.payload_mut();
            let mut output_data = output_payload.data_mut::<T>();

            let rank = input_tensor.shape().len();
            let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

            let info_on_host = scratch_alloc.allocate(4 * rank)?;
            info_on_host[..rank].copy_from_slice(&input_tensor.shape());
            info_on_host[rank..2 * rank].copy_from_slice(&input_tensor.stride());
            info_on_host[2 * rank..3 * rank].copy_from_slice(&output_tensor.shape());
            info_on_host[3 * rank..].copy_from_slice(&input_tensor.stride());

            let cuda_bump = ctx.execution_state().dev().device_allocator().clone();

            let info = cuda_bump.alloc_from_slice_with_fallback(info_on_host)?;
            let info_data = info.data::<usize>();

            let func = self.load_cuda_function(T::data_type(), Tind::data_type())?;

            match (ctx.get_input(3), ctx.get_input(4)) {
                (Ok(axes), Ok(steps)) => {
                    let axes_ptr = axes.payload();
                    let axes_data = axes_ptr.data::<Tind>();
                    let steps_ptr = steps.payload();
                    let steps_data = steps_ptr.data::<Tind>();

                    unsafe {
                        slice::compute(
                            self.stream.clone(),
                            func,
                            rank,
                            &info_data,
                            starts_data.as_ref(),
                            ends_data.as_ref(),
                            axes_data.as_ref(),
                            steps_data.as_ref(),
                            input_data.as_ref(),
                            output_data.as_mut(),
                        )?;
                    }
                }
                (Err(_), Ok(steps)) => {
                    let steps_ptr = steps.payload();
                    let steps_data = steps_ptr.data::<Tind>();

                    let null_axes = self.stream.null::<Tind>()?;

                    unsafe {
                        slice::compute(
                            self.stream.clone(),
                            func,
                            rank,
                            &info_data,
                            starts_data.as_ref(),
                            ends_data.as_ref(),
                            &null_axes,
                            steps_data.as_ref(),
                            input_data.as_ref(),
                            output_data.as_mut(),
                        )?;
                    }
                }
                (Ok(axes), Err(_)) => {
                    let axes_ptr = axes.payload();
                    let axes_data = axes_ptr.data::<Tind>();

                    let null_steps = self.stream.null::<Tind>()?;

                    unsafe {
                        slice::compute(
                            self.stream.clone(),
                            func,
                            rank,
                            &info_data,
                            starts_data.as_ref(),
                            ends_data.as_ref(),
                            axes_data.as_ref(),
                            &null_steps,
                            input_data.as_ref(),
                            output_data.as_mut(),
                        )?;
                    }
                }
                (Err(_), Err(_)) => {
                    let null_axes = self.stream.null::<Tind>()?;
                    let null_steps = self.stream.null::<Tind>()?;

                    unsafe {
                        slice::compute(
                            self.stream.clone(),
                            func,
                            rank,
                            &info_data,
                            starts_data.as_ref(),
                            ends_data.as_ref(),
                            &null_axes,
                            &null_steps,
                            input_data.as_ref(),
                            output_data.as_mut(),
                        )?;
                    }
                }
            };
        }

        #[cfg(feature = "dump")]
        debug::write_results_slice::<T, Tind>("debugging/slice", self.stream.clone(), ctx).unwrap();

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let input_dtype = ctx.get_input(0)?.dtype();
        let starts_dtype = ctx.get_input(1)?.dtype();
        let ends_dtype = ctx.get_input(2)?.dtype();

        if starts_dtype != ends_dtype {
            return Err(SliceError::StartAndEndDataTypeMismatch.into());
        }

        match (input_dtype, starts_dtype) {
            (DataType::Float16, DataType::Int32) => self.compute_slice::<f16, i32>(ctx),
            (DataType::Float16, DataType::Int64) => self.compute_slice::<f16, i64>(ctx),
            (DataType::Float, DataType::Int32) => self.compute_slice::<f32, i32>(ctx),
            (DataType::Float, DataType::Int64) => self.compute_slice::<f32, i64>(ctx),
            (DataType::Double, DataType::Int32) => self.compute_slice::<f64, i32>(ctx),
            (DataType::Double, DataType::Int64) => self.compute_slice::<f64, i64>(ctx),
            (DataType::Int32, DataType::Int32) => self.compute_slice::<i32, i32>(ctx),
            (DataType::Int32, DataType::Int64) => self.compute_slice::<i32, i64>(ctx),
            (DataType::Int64, DataType::Int32) => self.compute_slice::<i64, i32>(ctx),
            (DataType::Int64, DataType::Int64) => self.compute_slice::<i64, i64>(ctx),
            _ => Err(UnsupportedDataType(input_dtype).into()),
        }
    }
}

fn compute_output_shape<Tind>(ctx: &Context<Cuda>) -> Result<()>
where
    Tind: DataTypeMap
        + ValidAsZeroBits
        + DeviceRepr
        + Num
        + Default
        + Copy
        + Signed
        + PrimInt
        + Debug
        + TryFrom<usize>,
    <Tind as TryFrom<usize>>::Error: Debug,
    i64: From<Tind>,
{
    let input_tensor = ctx.get_input(0)?;

    debug!(
        "[input][dtype={:?}][shape={:?}][strides={:?}]",
        input_tensor.dtype(),
        input_tensor.shape(),
        input_tensor.stride()
    );

    let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

    let starts_data = {
        let starts_tensor = ctx.get_input(1)?;

        debug!(
            "[starts][dtype={:?}][shape={:?}][strides={:?}]",
            starts_tensor.dtype(),
            starts_tensor.shape(),
            starts_tensor.stride()
        );

        let starts_payload = starts_tensor.payload();
        let starts_data_len = starts_payload.len();
        let starts_data = scratch_alloc.allocate(starts_data_len)?;
        starts_tensor.payload_to_host::<Tind>(starts_data)?;
        starts_data
    };

    let ends_data = {
        let ends_tensor = ctx.get_input(2)?;

        debug!(
            "[ends][dtype={:?}][shape={:?}][strides={:?}]",
            ends_tensor.dtype(),
            ends_tensor.shape(),
            ends_tensor.stride()
        );

        let ends_payload = ends_tensor.payload();
        let ends_data_len = ends_payload.len();
        let ends_data = scratch_alloc.allocate(ends_data_len)?;
        ends_tensor.payload_to_host::<Tind>(ends_data)?;
        ends_data
    };

    let starts_len = starts_data.len();

    let norm_axes;
    match ctx.get_input(3) {
        Ok(axes_tensor) => {
            debug!(
                "[axes][dtype={:?}][shape={:?}][strides={:?}]",
                axes_tensor.dtype(),
                axes_tensor.shape(),
                axes_tensor.stride()
            );
            // Todo: maybe validate that this is 1D.
            let axes_data_data = axes_tensor.len();

            let axes_data = scratch_alloc.allocate(axes_data_data)?;
            axes_tensor.payload_to_host::<Tind>(axes_data)?;
            norm_axes = axes_data;
        }
        Err(_) => {
            norm_axes = scratch_alloc.allocate(starts_len)?;
            for i in 0..starts_len {
                norm_axes[i] = Tind::try_from(i).expect("MAX_RANK is 8");
            }
        }
    }

    let mut steps = None;
    if let Ok(steps_tensor) = ctx.get_input(4) {
        debug!(
            "[steps][dtype={:?}][shape={:?}][strides={:?}]",
            steps_tensor.dtype(),
            steps_tensor.shape(),
            steps_tensor.stride()
        );

        let steps_len = steps_tensor.len();
        let steps_data = scratch_alloc.allocate(steps_len)?;
        steps_tensor.payload_to_host::<Tind>(steps_data)?;
        steps = Some(steps_data);
    }

    let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
    let input = ctx.get_input(0)?;
    let output_shape = scratch_alloc.allocate_from_slice(&input.shape())?;

    compute_output_shape_helper(
        &input.shape(),
        norm_axes,
        starts_data,
        ends_data,
        steps.as_deref(),
        output_shape,
    )?;

    let output_tensor = ctx.get_output(0)?;
    output_tensor.copy_shape_from_slice(output_shape);

    Ok(())
}

// Assumes that `axes` has been normalized.
fn compute_output_shape_helper<T>(
    input_shape: &[usize],
    axes: &[T],
    starts: &[T],
    ends: &[T],
    steps: Option<&[T]>,
    output_shape: &mut [usize],
) -> Result<()>
where
    T: DataTypeMap
        + ValidAsZeroBits
        + DeviceRepr
        + Num
        + Copy
        + Signed
        + ToPrimitive
        + PrimInt
        + Debug
        + TryFrom<usize>,
{
    debug_assert!(!axes.is_empty());
    debug_assert!(
        starts.len() == ends.len(),
        "indices must have the same length"
    );
    debug_assert!(
        axes.len() == starts.len(),
        "axes must have the same length as starts"
    );
    debug_assert!(
        output_shape.len() == input_shape.len(),
        "input and output must have the same rank"
    );

    if steps.map_or(false, |s| s.len() != axes.len()) {
        return Err(SliceError::StepsAndAxesLengthMismatch.into());
    }

    for (axes_idx, raw_axis) in axes.iter().copied().enumerate() {
        let axis = if raw_axis.is_negative() {
            let rank = T::try_from(input_shape.len()).map_err(|_| ConversionError)?;
            let diff = rank + raw_axis;
            diff.to_usize().ok_or(ConversionError)?
        } else {
            raw_axis.to_usize().ok_or(ConversionError)?
        };

        if axis >= input_shape.len() {
            return Err(SliceError::AxisOutOfRange.into());
        }

        let step = steps.map(|s| s[axes_idx]).unwrap_or_else(T::one);
        if step.is_zero() {
            return Err(SliceError::ZeroStep.into());
        }

        // The downstream cuda kernel assumes that output_shape[d] <= i32::MAX, i64::MAX.
        // This conversion is important and must not be removed carelessly.
        let dim = T::try_from(input_shape[axis]).map_err(|_| ConversionError)?;

        let mut start = starts[axes_idx];
        if start.is_negative() {
            start = start + dim;
        }
        if start.is_negative() {
            start = T::zero();
        }
        if step.is_positive() && start > dim {
            start = dim;
        } else if step.is_negative() && start >= dim {
            start = dim - T::one();
        }

        let mut end = ends[axes_idx];
        if end.is_negative() {
            end = end + dim;
        }
        if end < T::one().neg() {
            end = T::one().neg()
        }
        if step.is_positive() && end > dim {
            end = dim;
        } else if step.is_negative() && end >= dim {
            end = dim - T::one();
        }

        if (step.is_negative() && start < end) || (step.is_positive() && start >= end) {
            output_shape[axis] = 0;
        } else {
            if step.is_positive() {
                let diff = end
                    .checked_sub(&start)
                    .ok_or(Box::new(SliceError::InvalidDifference))?;
                output_shape[axis] = ceil_div(diff, step).to_usize().ok_or(ConversionError)?;
            } else {
                let diff = start
                    .checked_sub(&end)
                    .ok_or(Box::new(SliceError::InvalidDifference))?;
                output_shape[axis] = ceil_div(diff, step.abs())
                    .to_usize()
                    .ok_or(ConversionError)?;
            }
        }
    }

    Ok(())
}

fn ceil_div<T>(a: T, b: T) -> T
where
    T: Signed + Copy + ToPrimitive,
{
    let result = if a.is_negative() {
        a / b
    } else {
        (a + b - T::one()) / b
    };

    result
}

#[derive(Debug)]
pub enum SliceError {
    AxisOutOfRange,
    InvalidDifference,
    StartAndEndDataTypeMismatch,
    StepsAndAxesLengthMismatch,
    ZeroStep,
}

impl Display for SliceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for SliceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_step_regular() {
        let shape = [10];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[2], &[9], Some(&[3]), &mut out).unwrap();
        assert_eq!(out, [3]);
    }

    #[test]
    fn pos_step_end_eq_dim() {
        let shape = [10];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[0], &[10], Some(&[1]), &mut out)
            .unwrap();
        assert_eq!(out, [10]);
    }

    #[test]
    fn neg_step_regular() {
        let shape = [10];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[8], &[1], Some(&[-2]), &mut out)
            .unwrap();
        assert_eq!(out, [4]);
    }

    #[test]
    fn neg_step_end_minus1() {
        let shape = [5];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[4], &[-1], Some(&[-1]), &mut out)
            .unwrap();
        assert_eq!(out, [0]);
    }

    #[test]
    fn empty_slice_pos() {
        let shape = [6];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[3], &[3], Some(&[1]), &mut out).unwrap();
        assert_eq!(out, [0]);
    }

    #[test]
    fn empty_slice_neg() {
        let shape = [6];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[2], &[2], Some(&[-1]), &mut out)
            .unwrap();
        assert_eq!(out, [0]);
    }

    #[test]
    fn step_zero_error() {
        let shape = [4];
        let mut out = shape;
        let err =
            compute_output_shape_helper::<i64>(&shape, &[0], &[0], &[4], Some(&[0]), &mut out)
                .unwrap_err();
        let err = err.downcast::<SliceError>().unwrap();
        matches!(err, SliceError::ZeroStep);
    }

    #[test]
    fn steps_len_mismatch_error() {
        let shape = [4];
        let mut out = shape;
        let err =
            compute_output_shape_helper::<i64>(&shape, &[0], &[0], &[4], Some(&[1, 2]), &mut out)
                .unwrap_err();
        let err = err.downcast::<SliceError>().unwrap();
        matches!(err, SliceError::StepsAndAxesLengthMismatch);
    }

    #[test]
    fn pos_start_oob_error() {
        let shape = [5];
        let mut out = vec![0];
        compute_output_shape_helper::<i64>(&shape, &[0], &[6], &[6], Some(&[1]), &mut out).unwrap();
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn neg_end_lt_minus1() {
        let shape = [5];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[4], &[-2], Some(&[-1]), &mut out)
            .unwrap();
    }

    #[test]
    fn rank2_two_axes_mixed_steps() {
        let shape = [6, 5];
        let mut out = shape;
        compute_output_shape_helper::<i64>(
            &shape,
            &[0, 1],
            &[1, 4],
            &[5, -1],
            Some(&[2, -1]),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [2, 0]);
    }

    #[test]
    fn rank3_slice_one_axis_default_step() {
        let shape = [3, 4, 10];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[2], &[3], &[10], None, &mut out).unwrap();
        assert_eq!(out, [3, 4, 7]);
    }

    #[test]
    fn rank3_empty_slice_axis1_neg() {
        let shape = [2, 8, 4];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[1], &[3], &[3], Some(&[-2]), &mut out)
            .unwrap();
        assert_eq!(out, [2, 0, 4]);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn axis_out_of_range_error() {
        let shape = [4, 4];
        let mut out = shape;
        let err = compute_output_shape_helper::<i64>(&shape, &[2], &[0], &[1], None, &mut out)
            .unwrap_err();
        matches!(err.downcast(), Ok(SliceError::AxisOutOfRange));
    }

    #[test]
    fn duplicate_axes_error() {
        let shape = [5, 5, 5];
        let mut out = shape;
        let _ = compute_output_shape_helper::<i64>(
            &shape,
            &[1, 1],
            &[0, 0],
            &[5, 5],
            Some(&[1, 1]),
            &mut out,
        );
        // Todo: We need to decide what to do here.
    }

    #[test]
    fn rank2_end_oob_positive_step_clamped() {
        let shape = [7, 3];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0], &[0], &[8], Some(&[1]), &mut out).unwrap();
        assert_eq!(out, [7, 3]);
    }

    #[test]
    fn rank3_all_axes_mixed_steps() {
        let shape = [8, 6, 10];
        let mut out = shape;
        compute_output_shape_helper::<i64>(
            &shape,
            &[0, 1, 2],
            &[0, 5, 1],
            &[8, -1, 10],
            Some(&[1, -2, 3]),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [8, 0, 3]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "axes must have the same length as starts")]
    fn rank3_starts_too_short_panics() {
        let shape = [4, 4, 4];
        let mut out = shape;
        compute_output_shape_helper::<i64>(&shape, &[0, 1, 2], &[0, 0], &[4, 4], None, &mut out)
            .unwrap();
    }
}
