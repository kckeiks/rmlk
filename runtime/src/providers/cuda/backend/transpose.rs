use crate::attributes::transpose;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct TransposeBackend {
    _stream: Arc<CudaStream>,
}

impl TransposeBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { _stream: stream }
    }

    fn compute_transpose<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        let input = ctx.get_input(0)?;

        debug!(
            "[input][shape={:?}][stride=[{:?}]",
            input.shape(),
            input.stride()
        );

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let output_shape = alloc.allocate(input.shape().len())?;

        match ctx
            .get_attributes()
            .as_ref()
            .and_then(|attrs| transpose::get_perm(&attrs))
        {
            None => {
                for (dim_i, dim) in input.shape().iter().rev().enumerate() {
                    output_shape[dim_i] = *dim;
                }
            }
            Some(perm) => {
                // The length of perm must be equal to the rank of the input.
                if perm.len() != output_shape.len() {
                    return Err(TransposeError::InvalidPermLength.into());
                }

                for (dst_i, dim_i) in perm.iter().enumerate() {
                    // Todo: add a more detailed error message.
                    let i =
                        usize::try_from(*dim_i).map_err(|_| InternalError::UnableToConvertValue)?;
                    if i >= output_shape.len() {
                        return Err(TransposeError::PermIndexOutOfBounds { index: *dim_i }.into());
                    }
                    output_shape[dst_i] = input.shape()[i];
                }
            }
        }
        let dst = ctx.get_output(0)?.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst)?;

        // Todo: to avoid problems with non-contiguous memory
        // for now we simply clone the data. We can do better.
        let input = ctx.get_input(0)?;
        let input_dev_data_ptr = input.try_dev_data_ptr()?;
        let dev_data = input_dev_data_ptr.data::<I>().clone();

        let mut output = ctx.get_output(0)?;

        debug!(
            "[output][shape={:?}][stride=[{:?}]",
            output.shape(),
            output.stride()
        );

        output.set_dev_data(CudaData::new(dev_data));

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_transpose::<f16>(ctx),
            DataType::Float => self.compute_transpose::<f32>(ctx),
            DataType::Double => self.compute_transpose::<f64>(ctx),
            DataType::Int32 => self.compute_transpose::<i32>(ctx),
            DataType::Int64 => self.compute_transpose::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

#[derive(Debug)]
pub enum TransposeError {
    InvalidPermLength,
    PermIndexOutOfBounds { index: i32 },
}

impl Display for TransposeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TransposeError::InvalidPermLength => {
                write!(f, "invalid perm length")
            }
            TransposeError::PermIndexOutOfBounds { index } => {
                write!(f, "perm index `{}` out of bounds", index)
            }
        }
    }
}

impl std::error::Error for TransposeError {}
