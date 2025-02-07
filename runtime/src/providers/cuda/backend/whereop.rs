use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct WhereBackend {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl WhereBackend {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
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
                    return Err(InternalError::IncompatibleTensorShape {
                        shapes: [
                            (x.src_id().into(), x.shape().to_vec()),
                            (y.src_id().into(), y.shape().to_vec()),
                        ]
                        .try_into()
                        .expect("Small map so should succeed"),
                        op: Op::Where,
                    });
                }

                let output_shape = alloc.allocate_fill(rank, 0)?;

                if !utils::compute_broadcast_output_shape(
                    inter_shape,
                    condition.shape(),
                    output_shape,
                ) {
                    return Err(InternalError::IncompatibleTensorShape {
                        shapes: [(condition.src_id().into(), y.shape().to_vec())]
                            .try_into()
                            .expect("Small map so should succeed"),
                        op: Op::Where,
                    });
                }

                let output = ctx.get_output(0)?;
                let output_index = output.dst_id();
                ctx.execution_state_mut()
                    .copy_shape_from_slice(output_shape, output_index)?;
            }
        }

        Ok(())
    }

    fn compute_where<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: WhereKernel,
    {
        // The output should have the same dimensions.
        // We do it now to avoid lifetime errors.
        self.process_shapes(ctx)?;

        let x = ctx.get_input(0)?;
        let y = ctx.get_input(1)?;
        let condition = ctx.get_input(2)?;

        {
            let output = ctx.get_output(0)?;
            debug!(
                "[x][where][shape={:?}][stride=[{:?}]",
                x.shape(),
                x.stride()
            );
            debug!(
                "[y][where][shape={:?}][stride=[{:?}]",
                y.shape(),
                y.stride()
            );
            debug!(
                "[condition][where][shape={:?}][stride=[{:?}]",
                condition.shape(),
                condition.stride()
            );
            debug!(
                "[output][where][shape={:?}][stride=[{:?}]",
                output.shape(),
                output.stride()
            );
        }

        let x_dev_data_ref = x.try_dev_data_ptr()?;
        let x_dev_data = x_dev_data_ref.data::<D>();

        let y_dev_data_ref = y.try_dev_data_ptr()?;
        let y_dev_data = y_dev_data_ref.data::<D>();

        let condition_dev_data_ref = condition.try_dev_data_ptr()?;
        let condition_dev_data = condition_dev_data_ref.data::<D>();

        let elem_count: usize = x.shape().iter().product();

        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let mut output = ctx.get_output(0)?;
            let output_dev_data_ref = output.dev_data_ptr_mut();
            let need_to_alloc_dev_data = output_dev_data_ref.is_none()
                || output_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != elem_count)
                    .unwrap_or(true);

            // We need to remove this immutable reference so we can mutate `y`.
            drop(output_dev_data_ref);

            if need_to_alloc_dev_data {
                let output_dev_data = self
                    .device
                    .alloc_zeros::<f32>(output.shape().iter().copied().product::<usize>())
                    .map_err(rmlk_cuda::Error::from)?;
                output.set_dev_data(CudaData::new(output_dev_data));
            };
        }

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

        T::execute::<D>(
            self.device,
            self.f,
            rank,
            info_buffer,
            &x_dev_data,
            &y_dev_data,
            &condition_dev_data,
            &mut output_dev_data,
        )?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: WhereKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();
        match dtype {
            DataType::Float => self.compute_where::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Where,
                dtype,
            }),
        }
    }
}

pub trait WhereKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        rank: usize,
        info: &[usize],
        x_dev_data: &CudaSlice<T>,
        y_dev_data: &CudaSlice<T>,
        condition_dev_data: &CudaSlice<T>,
        output_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl WhereKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        rank: usize,
        info: &[usize],
        x_dev_data: &CudaSlice<T>,
        y_dev_data: &CudaSlice<T>,
        condition_dev_data: &CudaSlice<T>,
        output_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        unsafe {
            rmlk_cuda::kernels::whereop::compute(
                device,
                func,
                rank,
                info,
                x_dev_data,
                y_dev_data,
                condition_dev_data,
                output_dev_data,
            )
            .map_err(Into::into)
        }
    }
}

pub struct NoOpKernel(());

impl WhereKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: CudaFunction,
        _: usize,
        _: &[usize],
        _: &CudaSlice<T>,
        _: &CudaSlice<T>,
        _: &CudaSlice<T>,
        _: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        Ok(())
    }
}
