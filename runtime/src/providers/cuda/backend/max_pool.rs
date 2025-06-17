use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct MaxPoolBackend {
    stream: Arc<CudaStream>,
}

impl MaxPoolBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl MaxPoolBackend {
    fn comput_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;

        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)?;
        let max_pool_attrs = MaxPoolAttributes::new(&attrs, ctx.execution_state().scratch_alloc())?;

        let mut y_shape = scratch_alloc.allocate_fill(x_shape.len(), 0)?;

        // Todo: Refactor function so we dont have to allocate a scratch buffer.
        // Todo: Move this to utils.
        // Todo: if attributes were usize, we wouldn't need to do this allocation here.
        rmlk_cuda::kernels::max_pool::compute_output_shape(
            &x_shape,
            max_pool_attrs.kernel_shape(),
            max_pool_attrs.pads(),
            max_pool_attrs.strides(),
            &mut y_shape,
            false,
        )?;

        let y = ctx.get_output(0)?;
        let y_index = y.dst_id();
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        ctx.execution_state_mut()
            .copy_shape_from_slice(shape, y_index)?;

        Ok(())
    }

    fn compute_max_pool<D, T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: MaxPoolKernel,
    {
        self.comput_output_shape(ctx)?;

        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;

        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)?;
        let max_pool_attrs = MaxPoolAttributes::new(&attrs, ctx.execution_state().scratch_alloc())?;

        debug!(
            "[x][max_pool][shape={:?}][stride=[{:?}]",
            x.shape(),
            x.stride()
        );
        debug!(
            "[y][max_pool][shape={:?}][stride=[{:?}]",
            y.shape(),
            y.stride()
        );
        debug!(
            "[max_pool][pads={:?}][strides=[{:?}][kernel_shape={:?}]",
            max_pool_attrs.pads(),
            max_pool_attrs.strides(),
            max_pool_attrs.kernel_shape()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc();

        let x_shape = scratch_alloc.allocate_and_convert_from_slice(x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(x.stride())?;
        let y_shape = scratch_alloc.allocate_and_convert_from_slice(y.shape())?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(y.stride())?;

        let x_dev_data_ref = x.try_dev_data_ptr()?;
        let x_dev_data = x_dev_data_ref.data();

        let elem_count = y_shape.iter().map(|n| *n as usize).product();

        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let mut y = ctx.get_output(0)?;
            let y_dev_data_ref = y.dev_data_ptr_mut();
            let need_to_alloc_dev_data = y_dev_data_ref.is_none()
                || y_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != elem_count)
                    .unwrap_or(true);

            // We need to remove this immutable reference so we can mutate `y`.
            drop(y_dev_data_ref);

            if need_to_alloc_dev_data {
                let y_dev_data = self
                    .stream
                    .alloc_zeros::<D>(elem_count)
                    .map_err(rmlk_cuda::Error::from)?;
                y.set_dev_data(CudaData::new(y_dev_data));
            };
        }

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        T::execute::<D>(
            self.stream.clone(),
            D::one(),
            D::zero(),
            &x_dev_data,
            &x_shape,
            &x_stride,
            max_pool_attrs.kernel_shape(),
            max_pool_attrs.pads(),
            max_pool_attrs.strides(),
            &mut y_dev_data,
            &y_shape,
            &y_stride,
        )?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: MaxPoolKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_max_pool::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedDataTypeForOp {
                op: Op::MaxPool,
                dtype,
            }),
        }
    }
}

pub trait MaxPoolKernel {
    fn execute<T>(
        stream: Arc<CudaStream>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl MaxPoolKernel for ActiveKernel {
    fn execute<T>(
        stream: Arc<CudaStream>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::max_pool::compute::<T>(
            stream,
            (alpha, beta),
            x_data,
            x_shape,
            x_stride,
            kernel_shape,
            pads,
            strides,
            y_data,
            y_shape,
            y_stride,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl MaxPoolKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaStream>,
        _: T,
        _: T,
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &mut CudaSlice<T>,
        _: &[i32],
        _: &[i32],
    ) -> Result<()> {
        Ok(())
    }
}
