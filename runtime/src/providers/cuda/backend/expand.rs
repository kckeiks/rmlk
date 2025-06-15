use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::core::Context;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_cuda::kernels::expand;
use rmlk_cuda::kernels::expand::ExpandKernel;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::cmp;
use std::sync::Arc;
use crate::providers::cuda::backend::common;

pub struct ExpandBackend {
    stream: Arc<CudaStream>,
}

impl ExpandBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => ExpandKernel::FwdF16,
            DataType::Float => ExpandKernel::FwdF32,
            DataType::Double => ExpandKernel::FwdF64,
            DataType::Int32 => ExpandKernel::FwdI32,
            _ => {
                return Err(InternalError::UnsupportedOpForDataType {
                    op: Op::Expand,
                    dtype,
                })
            }
        };

        expand::load_kernel(self.stream.context(), kernel_name).map_err(Into::into)
    }

    fn compute_expand<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(T::data_type())?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let output_shape = {
            let input_tensor = ctx.get_input(0)?;
            let shape_tensor = ctx.get_input(1)?;

            let rank = cmp::max(input_tensor.shape().len(), shape_tensor.shape().len());

            let shape_dev_ptr = shape_tensor.try_dev_data_ptr()?;
            let shape_view = shape_dev_ptr.data::<i64>();

            let shape_on_host = scratch_alloc.allocate::<i64>(shape_view.len())?;
            self.stream
                .memcpy_dtoh(shape_view.as_ref(), shape_on_host)?;
            let shape = scratch_alloc.allocate_and_convert_from_slice::<i64, usize>(shape_on_host)?;

            let output_shape = scratch_alloc.allocate(rank)?;

            if !utils::compute_broadcast_output_shape(input_tensor.shape(), shape, output_shape) {
                let a_id = input_tensor.src_id();
                let b_id = shape_tensor.src_id();
                return Err(InternalError::IncompatibleTensorShape {
                    shapes: [
                        (a_id.into(), input_tensor.shape().to_vec()),
                        (b_id.into(), shape_tensor.shape().to_vec()),
                    ]
                        .try_into()
                        .expect("Small map so should succeed"),
                    op: Op::Expand,
                });
            }
            
            output_shape
        };


        let output_tensor = ctx.get_output(0)?;
        let dst_id = output_tensor.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst_id)?;

        let output_tensor = ctx.get_output(0)?;
        common::init_tensor_device_data::<T>(&self.stream, output_tensor)?;

        let input_tensor = ctx.get_input(0)?;
        let shape_tensor = ctx.get_input(1)?;

        let rank = cmp::max(input_tensor.shape().len(), shape_tensor.shape().len());

        let input_dev_ptr = input_tensor.try_dev_data_ptr()?;
        let input_view = input_dev_ptr.data::<T>();
        
        let output_tensor = ctx.get_output(0)?;

        let input_rank =  input_tensor.shape().len();
        let output_rank =  output_tensor.shape().len();
        let info =
            scratch_alloc.allocate(2 * input_rank + 2 * output_rank)?;
        info[..input_rank].copy_from_slice(input_tensor.shape());
        info[input_rank..2 *input_rank].copy_from_slice(input_tensor.stride());
        info[2 * input_rank ..2 * input_rank + output_rank].copy_from_slice(output_tensor.shape());
        info[2 * input_rank + output_rank..].copy_from_slice(output_tensor.stride());

        let mut output_dev_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_dev_ptr.data_mut::<T>();

        unsafe {
            expand::compute(
                self.stream.clone(),
                func,
                rank,
                info,
                input_view.as_ref(),
                output_view.as_mut(),
            )?;
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_expand::<f32>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::ReduceMean,
                dtype,
            }),
        }
    }
}
