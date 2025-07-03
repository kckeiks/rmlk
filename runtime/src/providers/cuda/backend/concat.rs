use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::{attributes, utils};
use anyhow::Result;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct ConcatBackend {
    stream: Arc<CudaStream>,
}

impl ConcatBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn compute_output_shape(&mut self, ctx: &mut Context<Cuda>, target_axis: usize) -> Result<()> {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let input_tensor = ctx.get_input(0)?;
        let base_shape = scratch_alloc.allocate_from_slice(input_tensor.shape())?;

        let mut shape_on_axis = 0;
        for i in 0..usize::MAX {
            match ctx.get_input(i) {
                Ok(next_input_tensor) => {
                    if next_input_tensor.shape().len() != base_shape.len() {
                        return Err(ConcatError::ShapeMismatch {
                            index: i,
                            base_rank: base_shape.len(),
                            found_rank: next_input_tensor.shape().len(),
                        }
                        .into());
                    }

                    shape_on_axis += next_input_tensor.shape()[target_axis];

                    for (axis, dim) in next_input_tensor.shape().iter().enumerate() {
                        if axis == target_axis {
                            continue;
                        }

                        if *dim != base_shape[axis] {
                            return Err(ConcatError::DimensionMismatch {
                                axis,
                                dim: *dim,
                                expected_dim: base_shape[axis],
                            }
                            .into());
                        }
                    }
                }
                _ => break,
            }
        }

        base_shape[target_axis] = shape_on_axis;

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_slice(base_shape, dst_id)?;

        Ok(())
    }

    // TODO: If axis is last dimension (step == 1), consider larger DtoD copies
    fn compute_concat<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        {
            let attrs = ctx
                .get_attributes()
                .ok_or(InternalError::MissingAttributes)
                .map_err(Box::new)?;

            debug!("[attributes={:?}]", attrs);

            // Todo: validate axis.
            let raw_axis = attributes::concat::get_axis(&attrs).ok_or_else(|| {
                InternalError::MissingAttribute {
                    name: "`axis` is missing".to_string(),
                }
            })?;

            let input_tensor = ctx.get_input(0)?;

            if input_tensor.is_scalar() {
                return Err(ConcatError::ScalarsAreNotAllowed.into());
            }

            let axis = utils::normalize_index(raw_axis as i64, input_tensor.shape().len())?;

            self.compute_output_shape(ctx, axis)?;

            let output_tensor = ctx.get_output(0)?;

            debug!(
                "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
                output_tensor.dtype(),
                output_tensor.shape(),
                output_tensor.stride()
            );

            common::init_tensor_device_data::<I>(&self.stream, output_tensor)?;

            let output_tensor = ctx.get_output(0)?;
            let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
            let mut output_data = output_ptr.data_mut::<I>();

            let input = ctx.get_input(0)?;

            let outer_dims = input.shape()[..axis].iter().product::<usize>();
            let mut axis_offset = 0;
            for i in 0..usize::MAX {
                match ctx.get_input(i) {
                    Ok(input) => {
                        debug!(
                            "[input][{i}][dtype={:?}][shape={:?}][stride=[{:?}]",
                            input.dtype(),
                            input.shape(),
                            input.stride()
                        );

                        let dim = input.shape()[axis];
                        let step = input.stride()[axis];
                        let outer_block_size = step * dim;
                        let output_outer_block_size = output_tensor.shape()[axis] * step;

                        let input_ptr = input.try_dev_data_ptr()?;
                        let input_data = input_ptr.data::<I>();
                        for outer_i in 0..outer_dims {
                            for j in 0..dim {
                                let output_offset =
                                    (outer_i * output_outer_block_size) + (axis_offset + j) * step;

                                self.stream
                                    .memcpy_dtod(
                                        &input_data.slice(
                                            (outer_i * outer_block_size) + (j * step)
                                                ..(outer_i * outer_block_size) + (j * step) + step,
                                        ),
                                        &mut output_data
                                            .slice_mut(output_offset..output_offset + step),
                                    )
                                    .map_err(|e| InternalError::Device { error: e.into() })?;
                            }
                        }
                        axis_offset += dim;
                    }
                    _ => break,
                }
            }

            assert_eq!(
                output_data.len(),
                output_tensor.shape().iter().product::<usize>()
            );
        }

        #[cfg(feature = "debugger")]
        debug::write_results_concat::<I>("debugging/concat", self.stream.clone(), ctx)?;

        /*self.stream
        .synchronize()
        .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_concat::<f16>(ctx),
            DataType::Float => self.compute_concat::<f32>(ctx),
            DataType::Double => self.compute_concat::<f64>(ctx),
            DataType::Int32 => self.compute_concat::<i32>(ctx),
            DataType::Uint32 => self.compute_concat::<u32>(ctx),
            DataType::Int64 => self.compute_concat::<i64>(ctx),
            DataType::Uint64 => self.compute_concat::<u64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

#[derive(Debug)]
pub enum ConcatError {
    ShapeMismatch {
        index: usize,
        base_rank: usize,
        found_rank: usize,
    },
    DimensionMismatch {
        axis: usize,
        dim: usize,
        expected_dim: usize,
    },
    ScalarsAreNotAllowed,
}

impl Display for ConcatError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConcatError::ShapeMismatch { index, base_rank, found_rank } => write!(f, "shape mismatch for input {index}: base rank {base_rank} and found rank {found_rank}"),
            ConcatError::DimensionMismatch { axis, dim, expected_dim } => write!(f, "dimension mismatch for input {axis} and input dim {dim} expected dim {expected_dim}"),
            ConcatError::ScalarsAreNotAllowed => write!(f, "scalars are not allowed"),
        }
    }
}

impl std::error::Error for ConcatError {}
