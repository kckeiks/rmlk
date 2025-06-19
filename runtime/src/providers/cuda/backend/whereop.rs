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
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct WhereBackend {
    stream: Arc<CudaStream>,
    f: CudaFunction,
}

impl WhereBackend {
    pub fn new(stream: Arc<CudaStream>, f: CudaFunction) -> Self {
        Self { stream, f }
    }
}

impl WhereBackend {
    fn process_shapes(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let y = ctx.get_input(1)?;
        let condition = ctx.get_input(2)?;

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
        // The output should have the same dimensions.
        // We do it now to avoid lifetime errors.
        self.process_shapes(ctx)?;

        let x = ctx.get_input(0)?;

        debug!(
            "[x][where][shape={:?}][stride=[{:?}]",
            x.shape(),
            x.stride()
        );

        let y = ctx.get_input(1)?;

        debug!(
            "[y][where][shape={:?}][stride=[{:?}]",
            y.shape(),
            y.stride()
        );

        let condition = ctx.get_input(2)?;

        debug!(
            "[condition][where][shape={:?}][stride=[{:?}]",
            condition.shape(),
            condition.stride()
        );

        let x_dev_data_ref = x.try_dev_data_ptr()?;
        let x_dev_data = x_dev_data_ref.data::<D>();

        let y_dev_data_ref = y.try_dev_data_ptr()?;
        let y_dev_data = y_dev_data_ref.data::<D>();

        let condition_dev_data_ref = condition.try_dev_data_ptr()?;
        let condition_dev_data = condition_dev_data_ref.data::<D>();

        let output = ctx.get_output(0)?;

        debug!(
            "[output][where][shape={:?}][stride=[{:?}]",
            output.shape(),
            output.stride()
        );

        common::init_tensor_device_data::<D>(&self.stream, output)?;

        let output = ctx.get_output(0)?;
        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let x_stride = scratch_alloc.allocate_fill::<usize>(output.shape().len(), 0)?;
        utils::compute_broadcast_stride_from_output_shape(
            x.shape(),
            x.stride(),
            output.shape(),
            x_stride,
        );

        let y_stride = scratch_alloc.allocate_fill::<usize>(output.shape().len(), 0)?;
        utils::compute_broadcast_stride_from_output_shape(
            y.shape(),
            y.stride(),
            output.shape(),
            y_stride,
        );

        let condition_stride = scratch_alloc.allocate_fill::<usize>(output.shape().len(), 0)?;
        utils::compute_broadcast_stride_from_output_shape(
            condition.shape(),
            condition.stride(),
            output.shape(),
            condition_stride,
        );

        debug!("[x][where][broadcast][stride={:?}]", x_stride);
        debug!("[y][where][broadcast][stride={:?}]", y_stride);
        debug!(
            "[condition][where][broadcast][stride={:?}]",
            condition_stride
        );

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let output = ctx.get_output(0)?;
        let mut output_dev_data_ref = output.dev_data_ptr_mut();
        let mut output_dev_data = output_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        let rank = output.shape().len();

        let info_buffer = ctx.execution_state().scratch_alloc().allocate(4 * rank)?;
        info_buffer[..rank].copy_from_slice(output.shape());
        info_buffer[rank..2 * rank].copy_from_slice(x_stride);
        info_buffer[2 * rank..3 * rank].copy_from_slice(y_stride);
        info_buffer[3 * rank..].copy_from_slice(condition_stride);

        unsafe {
            rmlk_cuda::kernels::whereop::compute(
                self.stream,
                self.f,
                rank,
                info_buffer,
                &x_dev_data,
                &y_dev_data,
                &condition_dev_data,
                &mut output_dev_data,
            )?;
        }

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();
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
