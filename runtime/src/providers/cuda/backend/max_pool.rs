use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
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
            .ok_or(InternalError::MissingAttributes)
            .map_err(Box::new)?;
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

    fn compute_max_pool<D>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        self.comput_output_shape(ctx)?;

        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;

        debug!("[x][shape={:?}][stride=[{:?}]", x.shape(), x.stride());
        debug!("[y][shape={:?}][stride=[{:?}]", y.shape(), y.stride());

        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)
            .map_err(Box::new)?;
        let max_pool_attrs = MaxPoolAttributes::new(&attrs, ctx.execution_state().scratch_alloc())?;

        debug!(
            "[pads={:?}][strides=[{:?}][kernel_shape={:?}]",
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

        common::init_tensor_device_data::<D>(&self.stream, y)?;

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        rmlk_cuda::kernels::max_pool::compute::<D>(
            self.stream.clone(),
            (D::one(), D::zero()),
            &x_dev_data,
            x_shape,
            x_stride,
            max_pool_attrs.kernel_shape(),
            max_pool_attrs.pads(),
            max_pool_attrs.strides(),
            &mut y_dev_data,
            y_shape,
            y_stride,
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_max_pool::<f16>(ctx),
            DataType::Float => self.compute_max_pool::<f32>(ctx),
            DataType::Double => self.compute_max_pool::<f64>(ctx),
            DataType::Int32 => self.compute_max_pool::<i32>(ctx),
            DataType::Int64 => self.compute_max_pool::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
