use crate::attributes;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils::FromBytes;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::trilu;
use rmlk_cuda::kernels::trilu::TriluKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct TriluBackend {
    stream: Arc<CudaStream>,
}

impl TriluBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => TriluKernel::FwdF16,
            DataType::Float => TriluKernel::FwdF32,
            DataType::Double => TriluKernel::FwdF64,
            DataType::Int32 => TriluKernel::FwdI32,
            DataType::Uint32 => TriluKernel::FwdU32,
            DataType::Int64 => TriluKernel::FwdI64,
            DataType::Uint64 => TriluKernel::FwdU64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        trilu::load_kernel(self.stream.context(), kernel_name).map_err(Into::into)
    }

    fn get_k(&self, ctx: &Context<Cuda>) -> Result<i64> {
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        match ctx.get_input(1) {
            Ok(k_tensor) => {
                debug!(
                    "[k][shape={:?}][stride={:?}]",
                    k_tensor.shape(),
                    k_tensor.stride()
                );
                let k_dev_ptr = k_tensor.try_dev_data_ptr()?;
                let k_view = k_dev_ptr.data::<i64>();
                let k = scratch_alloc.allocate(1)?;
                self.stream
                    .memcpy_dtoh(k_view.as_ref(), k)
                    .map_err(|e| InternalError::Device { error: e.into() })?;
                Ok(k[0])
            }
            Err(_) => Ok(0),
        }
    }

    fn compute_trilu<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromBytes,
    {
        let func = self.load_cuda_function(T::data_type())?;

        let upper = match ctx.get_attributes() {
            None => true,
            Some(attr) => attributes::trilu::get_upper(&attr),
        };

        let input_tensor = ctx.get_input(0)?;

        debug!(
            "[input][shape={:?}][stride=[{:?}]",
            input_tensor.shape(),
            input_tensor.stride()
        );

        let output_tensor = ctx.get_output(0)?;
        let src_id = input_tensor.src_id();
        let dst_id = output_tensor.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_within(src_id, dst_id)?;

        let input_tensor = ctx.get_input(0)?;
        let rank = input_tensor.shape().len();
        let input_data_size = input_tensor.shape().iter().product::<usize>();

        let output_data = self
            .stream
            .alloc_zeros::<T>(input_data_size)
            .map_err(|e| InternalError::Device { error: e.into() })?;

        let mut output_tensor = ctx.get_output(0)?;

        debug!(
            "[output][shape={:?}][stride=[{:?}]",
            output_tensor.shape(),
            output_tensor.stride()
        );

        output_tensor.set_dev_data(CudaData::new(output_data));

        let k = self.get_k(ctx)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let info = scratch_alloc.allocate(2 * rank)?;
        info[..rank].copy_from_slice(input_tensor.shape());
        info[rank..].copy_from_slice(input_tensor.stride());

        let input_ptr = input_tensor.try_dev_data_ptr()?;
        let input_view = input_ptr.data::<T>();

        let mut output_ptr = output_tensor.try_dev_data_ptr_mut()?;
        let mut output_view = output_ptr.data_mut::<T>();

        unsafe {
            trilu::compute(
                self.stream.clone(),
                func,
                upper,
                k,
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
            DataType::Float16 => self.compute_trilu::<f16>(ctx),
            DataType::Float => self.compute_trilu::<f32>(ctx),
            DataType::Double => self.compute_trilu::<f64>(ctx),
            DataType::Int32 => self.compute_trilu::<i32>(ctx),
            DataType::Int64 => self.compute_trilu::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
