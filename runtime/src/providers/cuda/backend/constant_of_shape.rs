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

pub struct ConstantOfShape {
    device: Arc<CudaDevice>,
}

impl ConstantOfShape {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
        }
    }

    fn compute_constant_of_shape<O>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        O: Default + DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let input_data = {
            let input = ctx.get_input(0)?;
            let input_size = input.shape().iter().product();
            let on_host_data = scratch_alloc.allocate::<i64>(input_size)?;

            let data_ptr = input.try_dev_data_ptr()?;
            let dev_view = data_ptr.data::<i64>();

            self.device
                .dtoh_sync_copy_into(dev_view.as_ref(), on_host_data)?;

            on_host_data
        };

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();
        let output_shape = scratch_alloc.allocate_and_convert_from_slice(input_data)?;
        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst_id)?;

        let value = ctx
            .get_attributes()
            .and_then(attributes::constant_of_shape::get_value)
            .unwrap_or(0);
        if value == 0 {
            let output = ctx.get_output(0)?;
            common::init_tensor_device_data::<O>(&self.device, output)?;
        } else {
            let output = ctx.get_output(0)?;
            let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
            let on_host_data = scratch_alloc.allocate::<O>(output.shape().iter().product())?;

            let mut output_data_ptr = output.try_dev_data_ptr_mut()?;
            let mut output_data_view = output_data_ptr.data_mut::<O>();

            self.device
                .htod_sync_copy_into(on_host_data, output_data_view.as_mut())?;
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_constant_of_shape::<f32>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}

trait ConstantOfShapeProcessor {
    fn process<T>(&self, shape: &[T], value: T) -> Result<()>
    where
        T: Copy;
}
