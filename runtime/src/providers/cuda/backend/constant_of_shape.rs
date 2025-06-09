use crate::attributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct ConstantOfShapeBackend {
    stream: Arc<CudaStream>,
}

impl ConstantOfShapeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn compute_constant_of_shape<O>(&mut self, ctx: &mut Context<Cuda>, value: O) -> Result<()>
    where
        O: Copy + Default + DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let input_data = {
            let input = ctx.get_input(0)?;
            let input_size = input.shape().iter().product();
            let on_host_data = scratch_alloc.allocate::<i64>(input_size)?;

            let data_ptr = input.try_dev_data_ptr()?;
            let dev_view = data_ptr.data::<i64>();

            self.stream.memcpy_dtoh(dev_view.as_ref(), on_host_data)?;

            on_host_data
        };

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();
        let output_shape = scratch_alloc.allocate_and_convert_from_slice(input_data)?;
        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst_id)?;

        let output = ctx.get_output(0)?;
        common::init_tensor_device_data::<O>(&self.stream, output)?;

        if value == O::zero() {
            let output = ctx.get_output(0)?;
            let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
            let on_host_data =
                scratch_alloc.allocate_fill::<O>(output.shape().iter().product(), value)?;

            let mut output_data_ptr = output.try_dev_data_ptr_mut()?;
            let mut output_data_view = output_data_ptr.data_mut::<O>();

            self.stream
                .memcpy_htod(on_host_data, output_data_view.as_mut())?;
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx
            .get_attributes()
            .and_then(|attrs| attributes::constant_of_shape::get_dtype(&attrs))
            .unwrap_or(DataType::Float);

        match dtype {
            DataType::Float => {
                let value = ctx
                    .get_attributes()
                    .and_then(|attrs| attributes::constant_of_shape::get_value_f32(&attrs))
                    .unwrap_or(0.0);
                self.compute_constant_of_shape::<f32>(ctx, value)
            }
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::ConstantOfShape,
                dtype,
            }),
        }
    }
}
