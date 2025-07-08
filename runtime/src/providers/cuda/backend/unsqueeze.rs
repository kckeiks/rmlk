use crate::core::allocators::ScratchAllocator;
use crate::core::error::InternalError;
use crate::core::Context;

#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
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

    fn compute_unsqueeze<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let expanded_shape = compute_output_shape(&scratch_alloc, ctx)?;

        let data_tensor = ctx.get_input(0)?;
        let data_payload = data_tensor.payload();
        let data = data_payload.data::<T>();

        let expanded = ctx.get_output(0)?;
        expanded.copy_shape_from_slice(&expanded_shape);
        expanded.write_payload(&data)?;

        debug!(
            "[expanded][dtype={:?}][shape={:?}][stride={:?}]",
            expanded.dtype(),
            expanded.shape(),
            expanded.stride()
        );

        #[cfg(feature = "debugger")]
        debug::write_results_binary::<T, i64, T>(
            "debugging/unsqueeze",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_unsqueeze::<f16>(ctx),
            DataType::Float => self.compute_unsqueeze::<f32>(ctx),
            DataType::Double => self.compute_unsqueeze::<f64>(ctx),
            DataType::Int32 => self.compute_unsqueeze::<i32>(ctx),
            DataType::Int64 => self.compute_unsqueeze::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

fn compute_output_shape<'a>(
    scratch_alloc: &'a ScratchAllocator,
    ctx: &Context<Cuda>,
) -> Result<&'a [usize]> {
    let axes = ctx.get_input(1)?;

    debug!(
        "[axes][dtype={:?}][shape={:?}][stride=[{:?}]",
        axes.dtype(),
        axes.shape(),
        axes.stride()
    );

    if axes.shape().len() != 1 {
        return Err(UnsqueezeError::InvalidAxesRank.into());
    }

    let axes_value_count = *axes.shape().first().unwrap();
    let axes_data = scratch_alloc.allocate::<i64>(axes_value_count)?;
    axes.payload_to_host(axes_data)?;

    // Todo: validate the range of axes values.
    if duplicates_exist(&axes_data) {
        return Err(UnsqueezeError::DuplicateAxes.into());
    }

    let data = ctx.get_input(0)?;

    debug!(
        "[data][dtype={:?}][shape={:?}][stride=[{:?}]",
        data.dtype(),
        data.shape(),
        data.stride()
    );

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

#[derive(Debug)]
pub enum UnsqueezeError {
    InvalidAxesRank,
    DuplicateAxes,
}

impl Display for UnsqueezeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for UnsqueezeError {}
