use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::expand;
use rmlk_cuda::kernels::expand::ExpandKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::cmp;
use std::sync::Arc;

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
            DataType::Uint32 => ExpandKernel::FwdU32,
            DataType::Int64 => ExpandKernel::FwdI64,
            DataType::Uint64 => ExpandKernel::FwdU64,
            _ => {
                return Err(InternalError::UnsupportedDataType { dtype }.into());
            }
        };

        debug!("[kernel={:?}]", kernel_name);

        expand::load_kernel(self.stream.context(), kernel_name).map_err(Into::into)
    }

    fn compute_expand<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        {
            let func = self.load_cuda_function(T::data_type())?;

            let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

            let output_shape = {
                let input_tensor = ctx.get_input(0)?;

                debug!(
                    "[input][dtype={:?}][shape={:?}][stride={:?}]",
                    input_tensor.dtype(),
                    input_tensor.shape(),
                    input_tensor.stride()
                );

                let shape_tensor = ctx.get_input(1)?;

                debug!(
                    "[shape][dtype=i64][shape={:?}][stride={:?}]",
                    shape_tensor.shape(),
                    shape_tensor.stride()
                );

                let rank = shape_tensor.shape().iter().product();

                let shape_dev_ptr = shape_tensor.try_dev_data_ptr()?;
                let shape_view = shape_dev_ptr.data::<i64>();

                let shape_on_host = scratch_alloc.allocate::<i64>(shape_view.len())?;
                self.stream
                    .memcpy_dtoh(shape_view.as_ref(), shape_on_host)
                    .map_err(|e| InternalError::Device { error: e.into() })?;
                let target_shape =
                    scratch_alloc.allocate_and_convert_from_slice::<i64, usize>(shape_on_host)?;

                let output_shape = scratch_alloc.allocate(rank)?;

                if !utils::compute_broadcast_output_shape(
                    input_tensor.shape(),
                    target_shape,
                    output_shape,
                ) {
                    let a_id = input_tensor.src_id();
                    let b_id = shape_tensor.src_id();
                    return Err(InternalError::IncompatibleShapesForBroadcast {
                        shapes: [
                            (a_id.into(), input_tensor.shape().to_vec()),
                            (b_id.into(), shape_tensor.shape().to_vec()),
                        ]
                        .try_into()
                        .expect("Small map so should succeed"),
                    }
                    .into());
                }

                output_shape
            };

            let output_tensor = ctx.get_output(0)?;
            let dst_id = output_tensor.dst_id();
            ctx.execution_state_mut()
                .copy_shape_from_slice(output_shape, dst_id)?;

            let output_tensor = ctx.get_output(0)?;

            debug!(
                "[output][dtype={:?}][shape={:?}][stride={:?}]",
                output_tensor.dtype(),
                output_tensor.shape(),
                output_tensor.stride()
            );

            common::init_tensor_device_data::<T>(&self.stream, output_tensor)?;

            let input_tensor = ctx.get_input(0)?;
            let shape_tensor = ctx.get_input(1)?;

            // Todo: What is this unused?
            let rank = cmp::max(input_tensor.shape().len(), shape_tensor.shape().len());

            let input_dev_ptr = input_tensor.try_dev_data_ptr()?;
            let input_view = input_dev_ptr.data::<T>();

            let output_tensor = ctx.get_output(0)?;

            let input_rank = if !input_tensor.is_scalar() {
                input_tensor.shape().len()
            } else {
                1
            };

            let output_rank = output_tensor.shape().len();

            let info = match (input_tensor.is_scalar(), output_tensor.is_scalar()) {
                (false, false) => scratch_alloc.allocate(2 * input_rank + 2 * output_rank)?,
                (true, true) => scratch_alloc.allocate(4)?,
                (true, false) => scratch_alloc.allocate(2 + 2 * output_rank)?,
                (false, true) => scratch_alloc.allocate(2 + 2 * input_rank)?,
            };

            if input_tensor.is_scalar() {
                utils::write_info(&[1], &[1], info, 0);
            } else {
                utils::write_info(input_tensor.shape(), input_tensor.stride(), info, 0);
            }

            let start = if input_tensor.is_scalar() {
                2
            } else {
                2 * input_rank
            };

            let elem_count = output_tensor.shape().iter().product();

            if output_tensor.is_scalar() {
                utils::write_info(&[1], &[1], info, start);
            } else {
                utils::write_info(output_tensor.shape(), output_tensor.stride(), info, start);
            }

            let mut output_dev_ptr = output_tensor.try_dev_data_ptr_mut()?;
            let mut output_view = output_dev_ptr.data_mut::<T>();

            unsafe {
                expand::compute(
                    self.stream.clone(),
                    func,
                    input_rank,
                    output_rank,
                    info,
                    elem_count,
                    input_view.as_ref(),
                    output_view.as_mut(),
                )?;
            }
        }

        common::write_results_binary::<T, i64, T>(
            "debugging/expand",
            self.stream.clone(),
            ctx,
            Default::default(),
        )
        .unwrap();

        /*self.stream
            .synchronize()
            .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_expand::<f16>(ctx),
            DataType::Float => self.compute_expand::<f32>(ctx),
            DataType::Double => self.compute_expand::<f64>(ctx),
            DataType::Int32 => self.compute_expand::<i32>(ctx),
            DataType::Uint32 => self.compute_expand::<u32>(ctx),
            DataType::Int64 => self.compute_expand::<i64>(ctx),
            DataType::Uint64 => self.compute_expand::<u64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
