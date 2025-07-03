use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::whereop;
use rmlk_cuda::kernels::whereop::WhereKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct WhereBackend {
    stream: Arc<CudaStream>,
}

impl WhereBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl WhereBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => WhereKernel::WhereFwdF16,
            DataType::Float => WhereKernel::WhereFwdF32,
            DataType::Double => WhereKernel::WhereFwdF64,
            DataType::Int32 => WhereKernel::WhereFwdI32,
            DataType::Int64 => WhereKernel::WhereFwdI64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        whereop::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let condition = ctx.get_input(0)?;

        debug!(
            "[condition][dtype={:?}][shape={:?}][stride={:?}]",
            condition.dtype(),
            condition.shape(),
            condition.stride()
        );

        let x = ctx.get_input(1)?;

        debug!(
            "[x][dtype={:?}][shape={:?}][stride={:?}]",
            x.dtype(),
            x.shape(),
            x.stride()
        );

        let y = ctx.get_input(2)?;

        debug!(
            "[y][dtype={:?}][shape={:?}][stride={:?}]",
            y.dtype(),
            y.shape(),
            y.stride()
        );

        match x.shape() == y.shape() && x.shape() == condition.shape() {
            true => {
                let output = ctx.get_output(0)?;
                let x_index = x.src_id();
                let output_index = output.dst_id();
                ctx.execution_state_mut()
                    .copy_shape_from_within(x_index, output_index)?;
            }
            false => {
                let rank = [x.shape().len(), y.shape().len(), condition.shape().len()]
                    .into_iter()
                    .max()
                    .expect("Iterator is not empty");

                let alloc = ctx.execution_state().scratch_alloc().clone();
                let inter_shape = alloc.allocate_fill(rank, 0)?;

                if !utils::compute_broadcast_output_shape(x.shape(), y.shape(), inter_shape) {
                    return Err(InternalError::IncompatibleShapesForBroadcast {
                        shapes: [
                            (x.src_id().into(), x.shape().to_vec()),
                            (y.src_id().into(), y.shape().to_vec()),
                        ]
                        .try_into()
                        .expect("Small map so should succeed"),
                    }
                    .into());
                }

                let output_shape = alloc.allocate_fill(rank, 0)?;

                if !utils::compute_broadcast_output_shape(
                    inter_shape,
                    condition.shape(),
                    output_shape,
                ) {
                    return Err(InternalError::IncompatibleShapesForBroadcast {
                        shapes: [(condition.src_id().into(), y.shape().to_vec())]
                            .try_into()
                            .expect("Small map so should succeed"),
                    }
                    .into());
                }

                let output = ctx.get_output(0)?;
                let output_index = output.dst_id();
                ctx.execution_state_mut()
                    .copy_shape_from_slice(output_shape, output_index)?;
            }
        }

        Ok(())
    }

    fn compute_where<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        {
            let func = self.load_cuda_function(D::data_type())?;

            // The output should have the same dimensions.
            // We do it now to avoid lifetime errors.
            self.compute_output_shape(ctx)?;

            let condition = ctx.get_input(0)?;
            let x = ctx.get_input(1)?;
            let y = ctx.get_input(2)?;

            let x_dev_data_ref = x.try_dev_data_ptr()?;
            let x_dev_data = x_dev_data_ref.data::<D>();

            let y_dev_data_ref = y.try_dev_data_ptr()?;
            let y_dev_data = y_dev_data_ref.data::<D>();

            let condition_dev_data_ref = condition.try_dev_data_ptr()?;
            let condition_dev_data = condition_dev_data_ref.data::<bool>();

            let output = ctx.get_output(0)?;

            debug!(
                "[output][dtype={:?}][where][shape={:?}][stride=[{:?}]",
                output.dtype(),
                output.shape(),
                output.stride()
            );

            common::init_tensor_device_data::<D>(&self.stream, output)?;

            let output = ctx.get_output(0)?;
            let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

            let (x_shape, x_stride) = if x.is_scalar() {
                ([1].as_ref(), [1].as_ref())
            } else {
                (x.shape(), x.stride())
            };

            let output_shape = if output.is_scalar() {
                [1].as_ref()
            } else {
                output.shape()
            };

            let broadcast_x_stride = scratch_alloc.allocate_fill::<usize>(output_shape.len(), 0)?;
            utils::compute_broadcast_stride_from_output_shape(
                x_shape,
                x_stride,
                output_shape,
                broadcast_x_stride,
            );

            let (y_shape, y_stride) = if y.is_scalar() {
                ([1].as_ref(), [1].as_ref())
            } else {
                (y.shape(), y.stride())
            };

            let broadcast_y_stride = scratch_alloc.allocate_fill::<usize>(output_shape.len(), 0)?;
            utils::compute_broadcast_stride_from_output_shape(
                y_shape,
                y_stride,
                output_shape,
                broadcast_y_stride,
            );

            let (condition_shape, condition_stride) = if condition.is_scalar() {
                ([1].as_ref(), [1].as_ref())
            } else {
                (condition.shape(), condition.stride())
            };

            let broadcast_condition_stride =
                scratch_alloc.allocate_fill::<usize>(output_shape.len(), 0)?;
            utils::compute_broadcast_stride_from_output_shape(
                condition_shape,
                condition_stride,
                output_shape,
                broadcast_condition_stride,
            );

            debug!("[x][where][broadcast][stride={:?}]", broadcast_x_stride);
            debug!("[y][where][broadcast][stride={:?}]", broadcast_y_stride);
            debug!(
                "[condition][where][broadcast][stride={:?}]",
                broadcast_condition_stride
            );

            // The device data should exist so we will execute the kernel
            // and update the destination device data with the result.
            let output = ctx.get_output(0)?;
            let mut output_dev_data_ref = output.dev_data_ptr_mut();
            let mut output_dev_data = output_dev_data_ref
                .as_mut()
                .expect("we already checked that it initialized")
                .data_mut();

            let output_rank = output_shape.len();

            let info_buffer = ctx
                .execution_state()
                .scratch_alloc()
                .allocate(4 * output_rank)?;
            info_buffer[..output_rank].copy_from_slice(output_shape);
            info_buffer[output_rank..2 * output_rank].copy_from_slice(broadcast_x_stride);
            info_buffer[2 * output_rank..3 * output_rank].copy_from_slice(broadcast_y_stride);
            info_buffer[3 * output_rank..].copy_from_slice(broadcast_condition_stride);

            unsafe {
                whereop::compute(
                    self.stream.clone(),
                    func,
                    output_rank,
                    info_buffer,
                    &x_dev_data,
                    &y_dev_data,
                    &condition_dev_data,
                    &mut output_dev_data,
                )?;
            }
        }

        #[cfg(feature = "debugger")]
        debug::write_results_ternary::<bool, D, D, D>(
            "debugging/where",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        /*self.stream
        .synchronize()
        .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(1)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_where::<f16>(ctx),
            DataType::Float => self.compute_where::<f32>(ctx),
            DataType::Double => self.compute_where::<f64>(ctx),
            DataType::Int32 => self.compute_where::<i32>(ctx),
            DataType::Int64 => self.compute_where::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
