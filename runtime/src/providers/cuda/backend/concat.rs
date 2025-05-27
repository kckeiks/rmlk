use crate::attributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct ConcatBackend {
    device: Arc<CudaDevice>,
}

impl ConcatBackend {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
        }
    }

    fn compute_output_shape(&mut self, ctx: &mut Context<Cuda>, axis: usize) -> Result<()> {
        let input = ctx.get_input(0)?;
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let base_shape = scratch_alloc.allocate_from_slice(input.shape())?;

        for i in 0..usize::MAX {
            match ctx.get_input(i) {
                Ok(input) => {
                    if input.shape().len() != base_shape.len() {
                        return Err(InternalError::InvalidInput {
                            input: i,
                            op: Op::Concat,
                            message: "all inputs must have the same rank".to_string(),
                        });
                    }

                    base_shape[axis] += input.shape()[axis];

                    for (j, d) in input.shape().iter().enumerate() {
                        if j == axis {
                            continue;
                        }

                        if *d != base_shape[j] {
                            return Err(InternalError::InvalidInput {
                                input: j,
                                op: Op::Concat,
                                message: "dimensions do not match for axis `i`".to_string(),
                            });
                        }
                    }
                }
                _ => break,
            }
        }

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_slice(base_shape, dst_id)?;

        Ok(())
    }

    // TODO: If axis is last dimension (step == 1), consider larger DtoD copies
    fn compute_concat<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)?;
        let axis = usize::try_from(attributes::concat::get_axis(attrs).ok_or_else(|| {
            InternalError::MissingAttribute {
                name: "`axis` is missing".to_string(),
            }
        })?)
        .map_err(|_| InternalError::UnableToConvertValue)?;

        self.compute_output_shape(ctx, axis)?;

        let output_tensor = ctx.get_output(0)?;
        common::init_tensor_device_data::<I>(&self.device, output_tensor)?;

        let output_tensor = ctx.get_output(0)?;
        let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_data = output_ptr.data_mut::<I>();

        let input = ctx.get_input(0)?;
        let outer_dims = input.shape()[..axis].iter().product::<usize>();
        let mut axis_offset = 0;
        for i in 0..usize::MAX {
            match ctx.get_input(i) {
                Ok(input) => {
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

                            self.device.dtod_copy(
                                &input_data.slice(
                                    (outer_i * outer_block_size) + (j * step)
                                        ..(outer_i * outer_block_size) + (j * step) + step,
                                ),
                                &mut output_data.slice_mut(output_offset..output_offset + step),
                            )?;
                        }
                    }
                    axis_offset += dim;
                }
                _ => break,
            }
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_concat::<f32>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Range,
                dtype,
            }),
        }
    }
}
