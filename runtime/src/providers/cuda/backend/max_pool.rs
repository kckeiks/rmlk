use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::error::UnsupportedDataType;
use crate::core::Context;

use crate::attributes::error::AttributeError;
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
    fn compute_max_pool<T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        comput_output_shape(ctx)?;

        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;
        y.init_payload::<T>()?;

        debug!(
            "[y][dtype={:?}][shape={:?}][stride=[{:?}]",
            y.dtype(),
            y.shape(),
            y.stride()
        );

        let attrs = ctx
            .get_attributes()
            .ok_or(AttributeError::MissingAttributes)
            .map_err(Box::new)?;
        let max_pool_attrs = MaxPoolAttributes::new(&attrs, ctx.execution_state().scratch_alloc())?;

        debug!(
            "[pads={:?}][strides=[{:?}][kernel_shape={:?}]",
            max_pool_attrs.pads(),
            max_pool_attrs.strides(),
            max_pool_attrs.kernel_shape()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc();

        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;
        let y_shape = scratch_alloc.allocate_and_convert_from_slice(&y.shape())?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(&y.stride())?;

        let x_payload = x.payload();
        let x_data = x_payload.data();

        let y = ctx.get_output(0)?;
        let mut y_payload = y.payload_mut();
        let mut y_data = y_payload.data_mut();

        rmlk_cuda::kernels::max_pool::compute::<T>(
            self.stream.clone(),
            (T::one(), T::zero()),
            &x_data,
            x_shape,
            x_stride,
            max_pool_attrs.kernel_shape(),
            max_pool_attrs.pads(),
            max_pool_attrs.strides(),
            &mut y_data,
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
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn comput_output_shape(ctx: &Context<Cuda>) -> Result<()> {
    let x = ctx.get_input(0)?;

    debug!(
        "[x][dtype={:?}][shape={:?}][stride=[{:?}]",
        x.dtype(),
        x.shape(),
        x.stride()
    );

    let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
    let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;

    let attrs = ctx
        .get_attributes()
        .ok_or(AttributeError::MissingAttributes)
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
    let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
    y.copy_shape_from_slice(shape);

    Ok(())
}
