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

        // Todo: Validate that start and end values are clamped.
        self.compute_output_shape(norm_axes, starts_data, ends_data, steps.as_deref(), ctx)?;

        common::init_tensor_device_data::<T>(&self.device, ctx.get_output(0)?)?;

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
            let mut step_size = 1isize;
            let axis_slot = pos_in_axes(dim, axes);

            if let Some(slot) = axis_slot {
                if let Some(steps_sizes) = steps {
                    step_size = steps_sizes[slot]
                        .to_isize()
                        .ok_or(InternalError::UnableToConvertValue)?;
                    if step_size == 0 {
                        panic!("todo: throw error when step==0")
                    }
                }
            }

            coords[dim] += step_size;

            let end = match axis_slot {
                None => shape[dim]
                    .to_isize()
                    .ok_or(InternalError::UnableToConvertValue)?,
                Some(slot) => ends[slot]
                    .to_isize()
                    .ok_or(InternalError::UnableToConvertValue)?,
            };

            if (step_size > 0 && coords[dim] < end) || (step_size < 0 && coords[dim] > end) {
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
    debug_assert!(starts.len() == axes.len());
    debug_assert!(output_shape.len() > *axes.iter().max().unwrap_or(&0));

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

        if (step.is_negative() && starts[slot] <= ends[slot])
            || (step.is_positive() && starts[slot] >= ends[slot])
        {
            output_shape[axis] = 0;
        } else {
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
                    || end.to_usize().ok_or(InternalError::UnableToConvertValue)?
                        > input_shape[axis]
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
