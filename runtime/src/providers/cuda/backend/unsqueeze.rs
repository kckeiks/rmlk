use crate::core::allocators::ScratchAllocator;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::driver::CudaStream;
use rmlk_schema::{DataType, Op};
use std::sync::Arc;

pub struct UnsqueezeBackend {
    stream: Arc<CudaStream>,
}

impl UnsqueezeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn compute_output_shape<'a>(
        &self,
        scratch_alloc: &'a ScratchAllocator,
        ctx: &Context<Cuda>,
    ) -> Result<&'a [usize]> {
        let axes = ctx.get_input(1)?;

        if axes.shape().len() != 1 {
            return Err(InternalError::InvalidInput {
                input: 1,
                op: Op::NoOp,
                message: "`axes` is supposed to be a 1D tensor".to_string(),
            });
        }

        let axes_value_count = *axes.shape().first().unwrap();

        let axes_data = scratch_alloc.allocate::<i64>(axes_value_count)?;

        let axes_ptr = axes.try_dev_data_ptr()?;
        let axes_view = axes_ptr.data::<i64>();

        self.stream.memcpy_dtoh(axes_view.as_ref(), axes_data)?;

        // Todo: validate the range of axes values.
        if duplicates_exist(&axes_data) {
            return Err(InternalError::InvalidInput {
                input: 1,
                op: Op::Unsqueeze,
                message: format!("duplicates exist: {:?}", axes_data),
            });
        }

        let data = ctx.get_input(0)?;

        let expanded_rank = axes_value_count + data.shape().len();

        let expanded_shape = scratch_alloc.allocate_fill(expanded_rank, 0)?;

        for idx in axes_data {
            let norm_idx = utils::normalize_index(*idx, expanded_rank)?;
            expanded_shape[norm_idx] = 1;
        }

        let mut skip = 0;
        for idx in 0..expanded_shape.len() {
            if expanded_shape[idx] != 1 {
                expanded_shape[idx] = data.shape()[idx - skip];
            } else {
                skip += 1;
            }
        }

        Ok(expanded_shape)
    }

    fn compute_unsqueeze(&mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let expanded_shape = self.compute_output_shape(&scratch_alloc, ctx)?;

        let axes = ctx.get_input(1)?;
        let data_ptr = axes
            .dev_data_ptr_clone()
            .ok_or(InternalError::MissingDeviceData)?;

        let mut expanded = ctx.get_output(0)?;
        expanded.set_dev_data_ptr(data_ptr);

        let dst_id = expanded.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_slice(expanded_shape, dst_id)?;

        #[cfg(debug_assertions)]
        {
            use log::debug;

            let x = ctx.get_input(0)?;
            let y = ctx.get_output(0)?;
            debug!(
                "[x][flatten][shape={:?}][stride=[stride=[{:?}]",
                x.shape(),
                x.stride()
            );
            debug!(
                "[y][flatten][shape={:?}][stride=[stride=[{:?}]",
                y.shape(),
                y.stride()
            );
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float | DataType::Int64 => self.compute_unsqueeze(ctx),
            _ => Err(InternalError::UnsupportedDataTypeForOp {
                op: Op::Unsqueeze,
                dtype,
            }),
        }
    }
}

fn duplicates_exist<T: Eq>(slice: &[T]) -> bool {
    for i in 0..slice.len() {
        for j in (i + 1)..slice.len() {
            if slice[i] == slice[j] {
                return true;
            }
        }
    }
    false
}
