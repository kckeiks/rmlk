use crate::attributes;
use crate::core::allocators::ScratchAllocator;
use crate::core::error::InternalError;
use crate::core::Context;

#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::FromPrimitive;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct ReshapeBackend {
    _stream: Arc<CudaStream>,
}

impl ReshapeBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { _stream: stream }
    }

    fn compute_reshape<T>(&mut self, ctx: &Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let data_tensor = ctx.get_input(0)?;

        debug!(
            "[data][dtype={:?}][shape={:?}][stride={:?}]",
            data_tensor.dtype(),
            data_tensor.shape(),
            data_tensor.stride()
        );

        let target_shape = compute_output_shape(&scratch_alloc, ctx)?;

        let reshaped_tensor = ctx.get_output(0)?;
        reshaped_tensor.copy_shape_from_slice(&target_shape);
        reshaped_tensor.write_payload(&data_tensor.payload().data::<T>())?;

        debug!(
            "[reshaped][dtype={:?}][shape={:?}][stride={:?}]",
            reshaped_tensor.dtype(),
            reshaped_tensor.shape(),
            reshaped_tensor.stride()
        );

        #[cfg(feature = "dump")]
        debug::write_results_reshape::<T>("debugging/reshape", self.stream.clone(), ctx)?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_reshape::<f16>(ctx),
            DataType::Float => self.compute_reshape::<f32>(ctx),
            DataType::Double => self.compute_reshape::<f64>(ctx),
            DataType::Int32 => self.compute_reshape::<i32>(ctx),
            DataType::Int64 => self.compute_reshape::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

fn compute_output_shape<'a>(
    scratch_alloc: &'a ScratchAllocator,
    ctx: &Context<Cuda>,
) -> Result<&'a [usize]> {
    let allow_zero = match ctx.get_attributes() {
        Some(attrs) => {
            debug!("[attributes={attrs:?}]");
            attributes::reshape::get_allow_zero(&attrs)
        }
        None => false,
    };

    let shape_tensor = ctx.get_input(1)?;

    debug!(
        "[shape][dtype={:?}][shape={:?}][stride={:?}]",
        shape_tensor.dtype(),
        shape_tensor.shape(),
        shape_tensor.stride()
    );

    let shape_elem_count = if shape_tensor.shape().is_empty() {
        0
    } else {
        shape_tensor.shape().iter().product()
    };

    let dims_ints = scratch_alloc.allocate(shape_elem_count)?;
    shape_tensor.payload_to_host(dims_ints)?;

    let target_shape = scratch_alloc.allocate_fill::<usize>(dims_ints.len(), 0)?;

    let data_tensor = ctx.get_input(0)?;
    let data_shape = data_tensor.shape();
    let total_num_elems = data_shape.iter().product::<usize>();

    let mut neg_one_idx: Option<usize> = None;
    let mut has_zero_dim = false;

    for (idx, dim) in dims_ints.iter_mut().enumerate() {
        if *dim == -1 {
            if neg_one_idx.is_some() || (allow_zero && has_zero_dim) {
                return Err(ReshapeError::ZeroDimAndNegDimWhenAllowZero.into());
            }
            neg_one_idx = Some(idx);
            // Todo: This is ok because it has no effect in the multiplication we do later.
            target_shape[idx] = 1;
        } else if *dim == 0 {
            if allow_zero && neg_one_idx.is_some() {
                return Err(ReshapeError::ZeroDimAndNegDimWhenAllowZero.into());
            }

            has_zero_dim = true;

            if !allow_zero {
                target_shape[idx] = *data_shape.get(idx).ok_or(ReshapeError::IndexOutOfRange)?;
            } else {
                target_shape[idx] = 0;
            }
        } else {
            target_shape[idx] = usize::from_i64(*dim).ok_or(ReshapeError::InvalidIndex)?;
        }
    }

    if let Some(idx) = neg_one_idx {
        let cur_total_num_elems = target_shape.iter().product::<usize>();

        if cur_total_num_elems == 0 {
            if total_num_elems != 0 {
                return Err(ReshapeError::UnableToHoldSameNumberOfElements.into());
            }
            // Else, the input and output will be both scalars.
        } else {
            if total_num_elems % cur_total_num_elems != 0 {
                return Err(ReshapeError::UnableToHoldSameNumberOfElements.into());
            }

            let inferred_dim = total_num_elems / cur_total_num_elems;
            target_shape[idx] = inferred_dim;
        }
    }

    Ok(target_shape)
}

#[derive(Debug)]
pub enum ReshapeError {
    ZeroDimAndNegDimWhenAllowZero,
    IndexOutOfRange,
    UnableToHoldSameNumberOfElements,
    InvalidIndex,
}

impl Display for ReshapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReshapeError::ZeroDimAndNegDimWhenAllowZero => {
                write!(f, "ZeroDim and negative dimensions are not allowed")
            }
            ReshapeError::IndexOutOfRange => {
                write!(f, "index out of range")
            }
            ReshapeError::UnableToHoldSameNumberOfElements => {
                write!(f, "Unable to hold same number of elements")
            }
            ReshapeError::InvalidIndex => {
                write!(f, "Invalid index")
            }
        }
    }
}

impl std::error::Error for ReshapeError {}
