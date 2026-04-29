use crate::attributes;
use crate::attributes::scatter_nd::Reduction;
use crate::core::error::UnsupportedDataType;
use crate::core::Context;

#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::scatter_nd;
use rmlk_cuda::kernels::scatter_nd::ScatterNdKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct ScatterNdBackend {
    stream: Arc<CudaStream>,
}

impl ScatterNdBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, ctx: &Context<Cuda>, dtype: DataType) -> Result<CudaFunction> {
        let reduction = match ctx.get_attributes().as_ref() {
            Some(attrs) => attributes::scatter_nd::get_reduction(attrs)?,
            None => None,
        };

        debug!("[reduction={:?}]", reduction);

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
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        scatter_nd::load_kernel(self.stream.context(), kernel).map_err(Into::into)
    }

    fn compute_scatter_nd<T>(&mut self, ctx: &Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        {
            // Todo: can we preload the function when we need to read the attributes?
            let func = self.load_cuda_function(ctx, T::data_type())?;

            let data_tensor = ctx.get_input(0)?;

            debug!(
                "[data][dtype={:?}][shape={:?}][stride={:?}]",
                data_tensor.dtype(),
                data_tensor.shape(),
                data_tensor.stride()
            );

            if data_tensor.is_scalar() {
                return Err(ScatterNdError::ScalarInputsAreNotAllowed.into());
            }

            let indices_tensor = ctx.get_input(1)?;

            debug!(
                "[indices][dtype={:?}][shape={:?}][stride={:?}]",
                indices_tensor.dtype(),
                indices_tensor.shape(),
                indices_tensor.stride()
            );

            if indices_tensor.is_scalar() {
                return Err(ScatterNdError::ScalarInputsAreNotAllowed.into());
            }

            let updates_tensor = ctx.get_input(2)?;

            debug!(
                "[updates][dtype={:?}][shape={:?}][stride={:?}]",
                updates_tensor.dtype(),
                updates_tensor.shape(),
                updates_tensor.stride()
            );

            if updates_tensor.is_scalar() {
                return Err(ScatterNdError::ScalarInputsAreNotAllowed.into());
            }

            let output_tensor = ctx.get_output(0)?;
            output_tensor.copy_shape(data_tensor.shape_handle());

            let data_data = data_tensor.payload();
            output_tensor.write_payload(&data_data.data::<T>())?;

            debug!(
                "[output][dtype={:?}][shape={:?}][stride={:?}]",
                output_tensor.dtype(),
                output_tensor.shape(),
                output_tensor.stride()
            );

            let data_rank = data_tensor.shape().len();
            let indices_rank = indices_tensor.shape().len();
            let updates_rank = updates_tensor.shape().len();

            let num_idx_tuples = if indices_rank == 1 {
                // Todo: I think this can just be first().
                indices_tensor.shape().iter().product()
            } else {
                indices_tensor.shape()[0..indices_rank - 1].iter().product()
            };

            let indices_payload = indices_tensor.payload();
            let indices_data = indices_payload.data::<i64>();

            let updates_payload = updates_tensor.payload();
            let updates_data = updates_payload.data::<T>();

            let mut output_payload = output_tensor.payload_mut();
            let mut output_data = output_payload.data_mut::<T>();

            assert!(!output_data.is_empty());

            let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

            let info_on_host = scratch_alloc
                .allocate::<usize>(2 * data_rank + 2 * indices_rank + 2 * updates_rank)?;

            scatter_nd::create_info_buffer(
                &data_tensor.shape(),
                &data_tensor.stride(),
                &indices_tensor.shape(),
                &indices_tensor.stride(),
                &updates_tensor.shape(),
                &updates_tensor.stride(),
                info_on_host,
            );

            let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
            let info = cuda_bump.alloc_from_slice_with_fallback(info_on_host)?;
            let info_data = info.data::<usize>();

            let mut error = cuda_bump.alloc_with_fallback_zeroed::<i32>(1)?;
            let mut error_data = error.data_mut::<i32>();

            unsafe {
                scatter_nd::compute(
                    self.stream.clone(),
                    func,
                    num_idx_tuples,
                    data_rank,
                    indices_rank,
                    updates_rank,
                    &info_data,
                    indices_data.as_ref(),
                    updates_data.as_ref(),
                    output_data.as_mut(),
                    &mut error_data,
                )?;
            }

            let error_buf = scratch_alloc.allocate::<i32>(1)?;
            self.stream.memcpy_dtoh(error_data.as_ref(), error_buf)?;

            if error_buf[0] != 0 {
                return Err(ScatterNdError::KernelFailed { code: error_buf[0] }.into());
            }
        }

        #[cfg(feature = "dump")]
        debug::write_results_scatter_nd::<T>("debugging/scatter_nd", self.stream.clone(), ctx)?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_scatter_nd::<f16>(ctx),
            DataType::Float => self.compute_scatter_nd::<f32>(ctx),
            DataType::Double => self.compute_scatter_nd::<f64>(ctx),
            DataType::Int32 => self.compute_scatter_nd::<i32>(ctx),
            DataType::Uint32 => self.compute_scatter_nd::<u32>(ctx),
            DataType::Int64 => self.compute_scatter_nd::<i64>(ctx),
            DataType::Uint64 => self.compute_scatter_nd::<u64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScatterNdError {
    #[error("kernel failed with code {code}")]
    KernelFailed { code: i32 },
    #[error("scalar inputs are not allowed")]
    ScalarInputsAreNotAllowed,
}
