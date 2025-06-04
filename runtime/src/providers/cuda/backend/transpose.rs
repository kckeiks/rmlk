use crate::attributes::transpose;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, DeviceRepr, ValidAsZeroBits};
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct TransposeBackend {
    _device: Arc<CudaDevice>,
}

impl TransposeBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { _device: device }
    }

    fn compute_transpose<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        let input = ctx.get_input(0)?;

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
                    return Err(InternalError::InvalidAttribute {
                        name: "length of `perm` is not equal to rank of the tensor".to_string(),
                    });
                }

                for (dst_i, dim_i) in perm.iter().enumerate() {
                    // Todo: add a more detailed error message.
                    let i =
                        usize::try_from(*dim_i).map_err(|_| InternalError::UnableToConvertValue)?;
                    if i >= output_shape.len() {
                        return Err(InternalError::InvalidAttribute {
                            name: format!(
                                "index `{i}` is out of bounds for input shape `{:?}`",
                                input.shape()
                            ),
                        });
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

        output.set_dev_data(CudaData::new(dev_data));

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_transpose::<f32>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Transpose,
                dtype,
            }),
        }
    }
}
