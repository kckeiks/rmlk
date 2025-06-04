use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use num_traits::{Num, PrimInt, Signed, ToPrimitive};
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct SliceBackend {
    device: Arc<CudaDevice>,
}

impl SliceBackend {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
        }
    }

    fn compute_output_shape<Tind>(
        &mut self,
        axes: &[usize],
        starts: &[Tind],
        ends: &[Tind],
        steps: Option<&[Tind]>,
        ctx: &mut Context<Cuda>,
    ) -> Result<()>
    where
        Tind: DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + Copy
            + Signed
            + ToPrimitive
            + PrimInt,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let input = ctx.get_input(0)?;
        let output_shape = scratch_alloc.allocate_from_slice(input.shape())?;

        compute_output_shape(input.shape(), axes, starts, ends, steps, output_shape)?;

        let output_tensor = ctx.get_output(0)?;
        let dst_id = output_tensor.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst_id)?;

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn log_input_and_output(&mut self, ctx: &Context<Cuda>) {
        //use log::debug;
        todo!()
    }

    fn compute_slice<T, Tind>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num + Default + Copy,
        Tind: DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + Default
            + Copy
            + Signed
            + PrimInt,
        i64: From<Tind>,
    {
        let rank = ctx.get_input(0)?.shape().len();
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let norm_axes;
        match ctx.get_input(3) {
            Ok(axes_tensor) => {
                debug_assert_eq!(axes_tensor.dtype(), ctx.get_input(0)?.dtype());

                // Todo: maybe validate that this is 1D.
                let axes_ptr = axes_tensor.try_dev_data_ptr()?;
                let axes_view = axes_ptr.data::<Tind>();
                let axes_len = axes_view.len();

                let axes_data = scratch_alloc.allocate(axes_len)?;
                self.device
                    .dtoh_sync_copy_into(axes_view.as_ref(), axes_data)?;

                // Todo: how do we avoid this here? Maybe make the util function generic?
                let axes_data = scratch_alloc.allocate_and_convert_from_slice(axes_data)?;

                norm_axes = scratch_alloc.allocate(axes_len)?;
                utils::normalize_indices(axes_data, norm_axes, rank)?;
            }
            Err(_) => {
                norm_axes = scratch_alloc.allocate(rank)?;
                for i in 0..rank {
                    norm_axes[i] = i
                }
            }
        }

        let mut steps = None;
        if let Ok(steps_tensor) = ctx.get_input(4) {
            let steps_ptr = steps_tensor.try_dev_data_ptr()?;
            let steps_view = steps_ptr.data::<Tind>();
            let steps_len = steps_view.len();
            let steps_data = scratch_alloc.allocate(steps_len)?;
            self.device
                .dtoh_sync_copy_into(steps_view.as_ref(), steps_data)?;
            steps = Some(steps_data);
        }

        let starts_data = {
            let starts_tensor = ctx.get_input(1)?;

            let starts_ptr = starts_tensor.try_dev_data_ptr()?;
            let starts_view = starts_ptr.data::<Tind>();
            let starts_len = starts_view.len();

            let starts_data = scratch_alloc.allocate(starts_len)?;
            self.device
                .dtoh_sync_copy_into(starts_view.as_ref(), starts_data)?;

            starts_data
        };

        let ends_data = {
            let ends_tensor = ctx.get_input(2)?;

            let ends_ptr = ends_tensor.try_dev_data_ptr()?;
            let ends_view = ends_ptr.data::<Tind>();
            let ends_len = ends_view.len();

            let ends_data = scratch_alloc.allocate(ends_len)?;
            self.device
                .dtoh_sync_copy_into(ends_view.as_ref(), ends_data)?;

            ends_data
        };

        self.compute_output_shape(norm_axes, starts_data, ends_data, steps.as_deref(), ctx)?;

        if ctx.get_output(0)?.shape().iter().any(|&d| d == 0) {
            common::init_tensor_device_data_with_empty_slice::<T>(
                &self.device,
                ctx.get_output(0)?,
            )?;
            return Ok(());
        } else {
            common::init_tensor_device_data::<T>(&self.device, ctx.get_output(0)?)?;
        }

        let input_tensor = ctx.get_input(0)?;
        let input_ptr = input_tensor.try_dev_data_ptr()?;
        let input_view = input_ptr.data::<T>();

        let input_data = scratch_alloc.allocate::<T>(input_view.len())?;

        let output_data =
            scratch_alloc.allocate(ctx.get_output(0)?.shape().iter().product::<usize>())?;

        let coords_buf = scratch_alloc.allocate(rank)?;

        compute_slice(
            input_data,
            input_tensor.shape(),
            input_tensor.stride(),
            norm_axes,
            starts_data,
            ends_data,
            steps.as_deref(),
            coords_buf,
            output_data,
        )?;

        let output_tensor = ctx.get_output(0)?;
        let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_ptr.data_mut::<T>();

        self.device
            .htod_sync_copy_into(output_data, output_view.as_mut())?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let input_dtype = ctx.get_input(0)?.dtype();
        let starts_dtype = ctx.get_input(1)?.dtype();
        let ends_dtype = ctx.get_input(2)?.dtype();

        debug_assert_eq!(starts_dtype, ends_dtype);

        match (input_dtype, starts_dtype) {
            (DataType::Float, DataType::Int32) => self.compute_slice::<f32, i32>(ctx),
            (DataType::Float, DataType::Int64) => self.compute_slice::<f32, i64>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Range,
                dtype: input_dtype,
            }),
        }
    }
}

fn compute_slice<T, Tind>(
    input: &[T],
    shape: &[usize],
    strides: &[usize],
    axes: &[usize],
    starts: &[Tind],
    ends: &[Tind],
    steps: Option<&[Tind]>,
    coords: &mut [isize],
    output: &mut [T],
) -> Result<()>
where
    T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num + Copy,
    Tind: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num + Copy + ToPrimitive,
{
    // Todo: assert output is the right size.
    assert_eq!(axes.len(), starts.len());
    assert_eq!(axes.len(), ends.len());
    assert_eq!(axes.len(), steps.map(|s| s.len()).unwrap_or(axes.len()));

    let rank = shape.len();

    for dim in 0..rank {
        if let Some(axis_slot) = pos_in_axes(dim, axes) {
            coords[dim] = starts[axis_slot]
                .to_isize()
                .ok_or(InternalError::UnableToConvertValue)?;
        }
    }

    let mut output_offset = 0;
    loop {
        let mut input_offset = 0;

        for dim in 0..rank {
            input_offset += coords[dim]
                .to_usize()
                .ok_or(InternalError::UnableToConvertValue)?
                * strides[dim];
        }

        output[output_offset] = input[input_offset];
        output_offset += 1;

        let mut done = true;
        for dim in (0..rank).rev() {
            let mut step = 1isize;
            let axis_slot = pos_in_axes(dim, axes);

            if let Some(slot) = axis_slot {
                if let Some(steps_sizes) = steps {
                    step = steps_sizes[slot]
                        .to_isize()
                        .ok_or(InternalError::UnableToConvertValue)?;
                    if step == 0 {
                        panic!("todo: throw error when step==0")
                    }
                }
            }

            coords[dim] += step;

            let end = match axis_slot {
                None => shape[dim]
                    .to_isize()
                    .ok_or(InternalError::UnableToConvertValue)?,
                Some(slot) => ends[slot]
                    .to_isize()
                    .ok_or(InternalError::UnableToConvertValue)?,
            };

            if (step > 0 && coords[dim] < end) || (step < 0 && coords[dim] > end) {
                done = false;
                break;
            }

            coords[dim] = match axis_slot {
                None => 0,
                Some(idx) => starts[idx]
                    .to_isize()
                    .ok_or(InternalError::UnableToConvertValue)?,
            };
        }

        if done {
            break;
        }
    }

    Ok(())
}

fn pos_in_axes<T: Copy + PartialEq>(target: T, slice: &[T]) -> Option<usize> {
    slice.iter().position(|x| *x == target)
}

/// Computes the shape of the output tensor after Slice.
/// Requires axes, starts, ends (and steps, if present) to be the same length.
/// Panics in debug if axis ≥ output_shape.len().
fn compute_output_shape<T>(
    input_shape: &[usize],
    axes: &[usize],
    starts: &[T],
    ends: &[T],
    steps: Option<&[T]>,
    output_shape: &mut [usize],
) -> Result<()>
where
    T: DataTypeMap
        + CudnnDataType
        + ValidAsZeroBits
        + DeviceRepr
        + Num
        + Copy
        + Signed
        + ToPrimitive
        + PrimInt,
{
    debug_assert!(!axes.is_empty());
    debug_assert!(starts.len() <= axes.len());
    debug_assert!(
        output_shape.len() > *axes.iter().max().unwrap(),
        "axis values must be within range"
    );

    if steps.map_or(false, |s| s.len() != axes.len()) {
        // Todo: refactor errors.
        return Err(InternalError::InvalidInput {
            input: 4,
            op: Op::Slice,
            message: "`steps` length must match `axes`".into(),
        });
    }

    for (slot, axis) in axes.iter().copied().enumerate() {
        let step = steps.map(|s| s[slot]).unwrap_or_else(T::one);
        if step.is_zero() {
            return Err(InternalError::InvalidInput {
                input: 4,
                op: Op::Slice,
                message: "`step` cannot be zero".to_string(),
            });
        }

        let start = starts[slot];
        if step.is_positive() {
            if start < T::zero()
                || start
                    .to_usize()
                    .ok_or(InternalError::UnableToConvertValue)?
                    > input_shape[axis]
            {
                return Err(InternalError::UnableToConvertValue);
            }
        } else {
            if start < T::zero()
                || start
                    .to_usize()
                    .ok_or(InternalError::UnableToConvertValue)?
                    >= input_shape[axis]
            {
                return Err(InternalError::UnableToConvertValue);
            }
        }

        let end = ends[slot];
        if step.is_positive() {
            if end < T::zero()
                || end.to_usize().ok_or(InternalError::UnableToConvertValue)? > input_shape[axis]
            {
                return Err(InternalError::UnableToConvertValue);
            }
        } else {
            if !(end == T::one().neg()
                || (end >= T::zero()
                    && end.to_usize().ok_or(InternalError::UnableToConvertValue)?
                        < input_shape[axis]))
            {
                return Err(InternalError::UnableToConvertValue);
            }
        }

        if (step.is_negative() && start <= end) || (step.is_positive() && start >= end) {
            output_shape[axis] = 0;
        } else {
            if step.is_positive() {
                let diff = end
                    .checked_sub(&start)
                    .ok_or(InternalError::UnableToConvertValue)?;
                output_shape[axis] = ceil_div(diff, step)
                    .to_usize()
                    .ok_or(InternalError::UnableToConvertValue)?;
            } else {
                let diff = start
                    .checked_sub(&end)
                    .ok_or(InternalError::UnableToConvertValue)?;
                output_shape[axis] = ceil_div(diff, step.abs())
                    .to_usize()
                    .ok_or(InternalError::UnableToConvertValue)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Neg;

    fn range_tensor(len: usize) -> Vec<i32> {
        (0..len as i32).collect()
    }

    fn strides_for(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![0; shape.len()];
        utils::compute_stride::<usize>(shape, &mut s);
        s
    }

    #[test]
    fn pos_step_regular() {
        let shape = [10];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[0], &[2], &[9], Some(&[3]), &mut out).unwrap();
        assert_eq!(out, [3]);
    }

    #[test]
    fn pos_step_end_eq_dim() {
        let shape = [10];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[0], &[0], &[10], Some(&[1]), &mut out).unwrap();
        assert_eq!(out, [10]);
    }

    #[test]
    fn neg_step_regular() {
        let shape = [10];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[0], &[8], &[1], Some(&[-2]), &mut out).unwrap();
        assert_eq!(out, [4]);
    }

    #[test]
    fn neg_step_end_minus1() {
        let shape = [5];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[0], &[4], &[-1], Some(&[-1]), &mut out).unwrap();
        assert_eq!(out, [5]);
    }

    #[test]
    fn empty_slice_pos() {
        let shape = [6];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[0], &[3], &[3], Some(&[1]), &mut out).unwrap();
        assert_eq!(out, [0]);
    }

    #[test]
    fn empty_slice_neg() {
        let shape = [6];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[0], &[2], &[2], Some(&[-1]), &mut out).unwrap();
        assert_eq!(out, [0]);
    }

    #[test]
    fn step_zero_error() {
        let shape = [4];
        let mut out = shape;
        let err = compute_output_shape::<i64>(&shape, &[0], &[0], &[4], Some(&[0]), &mut out)
            .unwrap_err();
        matches!(err, InternalError::InvalidInput { .. });
    }

    #[test]
    fn steps_len_mismatch_error() {
        let shape = [4];
        let mut out = shape;
        let err = compute_output_shape::<i64>(&shape, &[0], &[0], &[4], Some(&[1, 2]), &mut out)
            .unwrap_err();
        matches!(err, InternalError::InvalidInput { .. });
    }

    #[test]
    fn pos_start_oob_error() {
        let shape = [5];
        let mut out = shape;
        let err = compute_output_shape::<i64>(&shape, &[0], &[6], &[6], Some(&[1]), &mut out)
            .unwrap_err();
        matches!(err, InternalError::UnableToConvertValue);
    }

    #[test]
    fn neg_end_lt_minus1_error() {
        let shape = [5];
        let mut out = shape;
        let err = compute_output_shape::<i64>(&shape, &[0], &[4], &[-2], Some(&[-1]), &mut out)
            .unwrap_err();
        matches!(err, InternalError::UnableToConvertValue);
    }

    #[test]
    fn rank2_two_axes_mixed_steps() {
        let shape = [6, 5];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[0, 1], &[1, 4], &[5, -1], Some(&[2, -1]), &mut out)
            .unwrap();
        assert_eq!(out, [2, 5]);
    }

    #[test]
    fn rank3_slice_one_axis_default_step() {
        let shape = [3, 4, 10];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[2], &[3], &[10], None, &mut out).unwrap();
        assert_eq!(out, [3, 4, 7]);
    }

    #[test]
    fn rank3_empty_slice_axis1_neg() {
        let shape = [2, 8, 4];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[1], &[3], &[3], Some(&[-2]), &mut out).unwrap();
        assert_eq!(out, [2, 0, 4]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "axis values must be within range")]
    fn axis_out_of_range_error() {
        let shape = [4, 4];
        let mut out = shape;
        compute_output_shape::<i64>(&shape, &[2], &[0], &[1], None, &mut out).unwrap();
    }

    #[test]
    fn duplicate_axes_error() {
        let shape = [5, 5, 5];
        let mut out = shape;
        let err =
            compute_output_shape::<i64>(&shape, &[1, 1], &[0, 0], &[5, 5], Some(&[1, 1]), &mut out);
        // Todo: We need to decide what to do here.
    }

    #[test]
    fn rank2_end_oob_positive_step_error() {
        let shape = [7, 3];
        let mut out = shape;
        let err = compute_output_shape::<i64>(&shape, &[0], &[0], &[8], Some(&[1]), &mut out)
            .unwrap_err();
        matches!(err, InternalError::UnableToConvertValue);
    }

    #[test]
    fn rank3_all_axes_mixed_steps() {
        let shape = [8, 6, 10];
        let mut out = shape;
        compute_output_shape::<i64>(
            &shape,
            &[0, 1, 2],
            &[0, 5, 1],
            &[8, -1, 10],
            Some(&[1, -2, 3]),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [8, 3, 3]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn rank3_starts_too_short_panics() {
        let shape = [4, 4, 4];
        let mut out = shape;
        let _ = compute_output_shape::<i64>(&shape, &[0, 1, 2], &[0, 0], &[4, 4], None, &mut out);
    }

    #[test]
    fn rank1_step1() {
        let shape = [10];
        let input = range_tensor(shape.iter().product());
        let strides = strides_for(&shape);
        let mut coords = [0isize];
        let mut out = vec![0; 6];

        compute_slice::<i32, i64>(
            &input,
            &shape,
            &strides,
            &[0],
            &[2],
            &[8],
            None,
            &mut coords,
            &mut out,
        )
        .unwrap();

        assert_eq!(&out, &[2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn rank1_step2_neg() {
        let shape = [10];
        let input = range_tensor(shape.iter().product());
        let strides = strides_for(&shape);
        let mut coords = [0isize];
        let mut out = vec![0; 3];

        compute_slice::<i32, i64>(
            &input,
            &shape,
            &strides,
            &[0],
            &[7],
            &[1],
            Some(&[-2]),
            &mut coords,
            &mut out,
        )
        .unwrap();

        assert_eq!(&out, &[7, 5, 3]);
    }

    #[test]
    fn rank2_mixed_steps() {
        let shape = [3, 4];
        let input = range_tensor(shape.iter().product());
        let strides = strides_for(&shape);
        let mut coords = [0isize; 2];
        let mut out = vec![0; 4];

        compute_slice::<i32, i64>(
            &input,
            &shape,
            &strides,
            &[0, 1],
            &[1, 1],
            &[3, 4],
            Some(&[1, -2i64.neg()]),
            &mut coords,
            &mut out,
        )
        .unwrap();

        assert_eq!(&out, &[5, 7, 9, 11]);
    }

    #[test]
    fn rank3_slice_last_axis() {
        // shape (2,2,3) keep axis-2 1..3
        let shape = [2, 2, 3];
        let input = range_tensor(shape.iter().product());
        let strides = strides_for(&shape);
        let mut coords = [0isize; 3];
        let mut out = vec![0; 8];

        compute_slice::<i32, i64>(
            &input,
            &shape,
            &strides,
            &[2],
            &[1],
            &[3],
            None,
            &mut coords,
            &mut out,
        )
        .unwrap();

        assert_eq!(&out, &[1, 2, 4, 5, 7, 8, 10, 11]);
    }

    #[test]
    #[should_panic]
    fn step_zero_panics() {
        let shape = [5];
        let input = range_tensor(shape.iter().product());
        let strides = strides_for(&shape);
        let mut coords = [0isize];
        let mut out = vec![0; 1];

        let _ = compute_slice::<i32, i64>(
            &input,
            &shape,
            &strides,
            &[0],
            &[0],
            &[5],
            Some(&[0]),
            &mut coords,
            &mut out,
        );
    }

    #[test]
    #[should_panic]
    fn steps_len_mismatch_panics() {
        let shape = [5];
        let input = range_tensor(shape.iter().product());
        let strides = strides_for(&shape);
        let mut coords = [0isize];
        let mut out = vec![0; 1];

        let _ = compute_slice::<i32, i64>(
            &input,
            &shape,
            &strides,
            &[0],
            &[0],
            &[5],
            Some(&[1, 2]),
            &mut coords,
            &mut out,
        );
    }
}
