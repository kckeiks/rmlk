use crate::attributes;
use crate::attributes::scatter_nd::Reduction;
use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_cuda::kernels::scatter_nd;
use rmlk_cuda::kernels::scatter_nd::ScatterNdKernel;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;
use log::debug;

pub struct ScatterNdBackend {
    stream: Arc<CudaStream>,
}

impl ScatterNdBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, ctx: &mut Context<Cuda>, dtype: DataType) -> Result<CudaFunction> {
        let reduction = match ctx.get_attributes().as_ref() {
            Some(attrs) => attributes::scatter_nd::get_reduction(attrs)?,
            None => None,
        };

        let kernel = match reduction {
            None if matches!(dtype, DataType::Float16) => ScatterNdKernel::FwdF16,
            None if matches!(dtype, DataType::Float) => ScatterNdKernel::FwdF32,
            None if matches!(dtype, DataType::Double) => ScatterNdKernel::FwdF64,
            None if matches!(dtype, DataType::Int32) => ScatterNdKernel::FwdI32,
            None if matches!(dtype, DataType::Int64) => ScatterNdKernel::FwdI64,
            Some(Reduction::Add) if matches!(dtype, DataType::Float16) => {
                ScatterNdKernel::AddFwdF16
            }
            Some(Reduction::Add) if matches!(dtype, DataType::Float) => ScatterNdKernel::AddFwdF32,
            Some(Reduction::Add) if matches!(dtype, DataType::Double) => ScatterNdKernel::AddFwdF64,
            Some(Reduction::Add) if matches!(dtype, DataType::Int32) => ScatterNdKernel::AddFwdI32,
            Some(Reduction::Add) if matches!(dtype, DataType::Int64) => ScatterNdKernel::AddFwdI64,
            Some(Reduction::Mul) if matches!(dtype, DataType::Float16) => {
                ScatterNdKernel::MulFwdF16
            }
            Some(Reduction::Mul) if matches!(dtype, DataType::Float) => ScatterNdKernel::MulFwdF32,
            Some(Reduction::Mul) if matches!(dtype, DataType::Double) => ScatterNdKernel::MulFwdF64,
            Some(Reduction::Mul) if matches!(dtype, DataType::Int32) => ScatterNdKernel::MulFwdI32,
            Some(Reduction::Mul) if matches!(dtype, DataType::Int64) => ScatterNdKernel::MulFwdI64,
            Some(Reduction::Max) if matches!(dtype, DataType::Float16) => {
                ScatterNdKernel::MaxFwdF16
            }
            Some(Reduction::Max) if matches!(dtype, DataType::Float) => ScatterNdKernel::MaxFwdF32,
            Some(Reduction::Max) if matches!(dtype, DataType::Double) => ScatterNdKernel::MaxFwdF64,
            Some(Reduction::Max) if matches!(dtype, DataType::Int32) => ScatterNdKernel::MaxFwdI32,
            Some(Reduction::Max) if matches!(dtype, DataType::Int64) => ScatterNdKernel::MaxFwdI64,
            Some(Reduction::Min) if matches!(dtype, DataType::Float16) => {
                ScatterNdKernel::MinFwdF16
            }
            Some(Reduction::Min) if matches!(dtype, DataType::Float) => ScatterNdKernel::MinFwdF32,
            Some(Reduction::Min) if matches!(dtype, DataType::Double) => ScatterNdKernel::MinFwdF64,
            Some(Reduction::Min) if matches!(dtype, DataType::Int32) => ScatterNdKernel::MinFwdI32,
            Some(Reduction::Min) if matches!(dtype, DataType::Int64) => ScatterNdKernel::MinFwdI64,
            _ => {
                return Err(InternalError::UnsupportedOpForDataType {
                    op: Op::ScatterND,
                    dtype,
                })
            }
        };

        scatter_nd::load_kernel(self.stream.context(), kernel).map_err(Into::into)
    }

    fn compute_scatter_nd<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        // Todo: can we preload the function when we need to read the attributes?
        let func = self.load_cuda_function(ctx, T::data_type())?;

        {
            let data_tensor = ctx.get_input(0)?;
            let output_tensor = ctx.get_output(0)?;
            let src_id = data_tensor.src_id();
            let dst_id = output_tensor.dst_id();
            ctx.execution_state_mut()
                .copy_shape_from_within(src_id, dst_id)?;

            // We copy `data` into the output tensor.
            let data_tensor = ctx.get_input(0)?;
            let mut output_tensor = ctx.get_output(0)?;
            common::copy_tensor_dev_data::<T>(&self.stream, &data_tensor, &mut output_tensor)?;
        }

        let data_tensor = ctx.get_input(0)?;
        let indices_tensor = ctx.get_input(1)?;
        let updates_tensor = ctx.get_input(2)?;

        let data_rank = data_tensor.shape().len();
        let indices_rank = indices_tensor.shape().len();
        let updates_rank = updates_tensor.shape().len();

        let num_idx_tuples = if indices_rank == 1 {
            // Todo: I think this can just be first().
            indices_tensor.shape().iter().product()
        } else {
            indices_tensor.shape()[0..indices_rank - 1].iter().product()
        };

        let indices_dev_ptr = indices_tensor.try_dev_data_ptr()?;
        let indices_view = indices_dev_ptr.data::<i64>();

        let updates_dev_ptr = updates_tensor.try_dev_data_ptr()?;
        let updates_view = updates_dev_ptr.data::<T>();

        let output_tensor = ctx.get_output(0)?;
        let mut output_dev_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_dev_ptr.data_mut::<T>();

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let info =
            scratch_alloc.allocate::<usize>(2 * data_rank + 2 * indices_rank + 2 * updates_rank)?;
        scatter_nd::create_info_buffer(
            data_tensor.shape(),
            data_tensor.stride(),
            indices_tensor.shape(),
            indices_tensor.stride(),
            updates_tensor.shape(),
            updates_tensor.stride(),
            info,
        );

        // Todo: maybe we should preallocate this value since its size never changes.
        let mut error = self.stream.alloc_zeros::<i32>(1)?;
        
        #[cfg(debug_assertions)]
        {
            debug!("num_idx_tuple=num_idx_tuples={num_idx_tuples}");
            debug!("data_rank={num_idx_tuples}");
            debug!("indices_rank={indices_rank}");
            debug!("updates_rank={updates_rank}");
            debug!("info={info:?}");
            debug!("indices size={}", indices_view.len());
            debug!("updates size={}", updates_view.len());
            debug!("output size={}", output_view.len());  
        }

        unsafe {
            scatter_nd::compute(
                self.stream.clone(),
                func,
                num_idx_tuples,
                data_rank,
                indices_rank,
                updates_rank,
                &info,
                indices_view.as_ref(),
                updates_view.as_ref(),
                output_view.as_mut(),
                &mut error,
            )?;
        }

        let error_buf = scratch_alloc.allocate::<i32>(1)?;
        self.stream.memcpy_dtoh(&error, error_buf)?;

        if error_buf[0] != 0 {
            return Err(InternalError::InvalidInput {
                input: 0,
                op: Op::ScatterND,
                message: format!("Scatter ND error: {}", error_buf[0]),
            })
        }

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_scatter_nd::<f32>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::ReduceMean,
                dtype,
            }),
        }
    }
}
