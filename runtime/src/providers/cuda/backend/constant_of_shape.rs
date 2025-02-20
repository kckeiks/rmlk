use std::sync::Arc;
use cudarc::driver::CudaDevice;
use rmlk_schema::{DataType, Op};
use crate::attributes;
use crate::core::Context;
use crate::core::error::{InternalError, Result};
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;

pub struct ConstantOfShape {
    device: Arc<CudaDevice>
}

impl ConstantOfShape {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
        }
    }
    
    fn compute_constant_of_shape<O>(&mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let input = ctx.get_input(0)?;
        
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let on_host_data = scratch_alloc.allocate::<i64>(input.shape().iter().product())?;
        
        let data_ptr = input.try_dev_data_ptr()?;
        let dev_view = data_ptr.data::<i64>();
        
        self.device.dtod_copy(&dev_view, on_host_data).map_err(Into::into)?;

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();
        let output_shape = ctx.execution_state().scratch_alloc().allocate_and_convert_from_slice(on_host_data)?;
        ctx.execution_state_mut().copy_shape_from_slice(output_shape, dst_id)?;

        
        let output = ctx.get_output(0)?;
        common::init_tensor_device_data(&self.device, output)?;
        

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()>
    {
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
        T: Copy
    ;
}