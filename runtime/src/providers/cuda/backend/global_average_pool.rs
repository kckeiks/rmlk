use crate::core::error::InternalError;
use crate::core::Context;

use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct GlobalAverageBackend {
    stream: Arc<CudaStream>,
}

impl GlobalAverageBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl GlobalAverageBackend {
    fn compute_global_average_pool<T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        comput_output_shape(ctx)?;

        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;
        y.init_payload::<T>()?;

        debug!(
            "[y][dtype={:?}][global_avg_pool][shape={:?}][stride=[{:?}]",
            y.dtype(),
            y.shape(),
            y.stride()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc();

        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;

        let y_shape = scratch_alloc.allocate_and_convert_from_slice(&y.shape())?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(&y.stride())?;

        let pads = scratch_alloc.allocate_fill(x_shape[2..].len(), 0)?;
        let strides = scratch_alloc.allocate_fill(x_shape[2..].len(), 1)?;
        let kernel_shape = &x_shape[2..];

        debug!(
            "[global_avg_pool][pads={:?}][strides=[{:?}][kernel_shape={:?}]",
            pads, strides, kernel_shape
        );

        let x_payload = x.payload();
        let x_data = x_payload.data();

        let y = ctx.get_output(0)?;
        let mut y_payload = y.payload_mut();
        let mut y_dev_data = y_payload.data_mut();

        rmlk_cuda::kernels::global_average_pool::compute::<T>(
            self.stream.clone(),
            (T::one(), T::zero()),
            pads,
            strides,
            &x_data,
            x_shape,
            x_stride,
            kernel_shape,
            &mut y_dev_data,
            y_shape,
            y_stride,
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_global_average_pool::<f16>(ctx),
            DataType::Float => self.compute_global_average_pool::<f32>(ctx),
            DataType::Double => self.compute_global_average_pool::<f64>(ctx),
            DataType::Int32 => self.compute_global_average_pool::<i32>(ctx),
            DataType::Int64 => self.compute_global_average_pool::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

fn comput_output_shape(ctx: &mut Context<Cuda>) -> Result<()> {
    let x = ctx.get_input(0)?;

    debug!(
        "[x][dtype={:?}][global_avg_pool][shape={:?}][stride=[{:?}]",
        x.dtype(),
        x.shape(),
        x.stride()
    );

    let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

    let y_shape_original = scratch_alloc.allocate_fill(x.shape().len(), 0)?;
    rmlk_cuda::kernels::global_average_pool::compute_output_shape(&x.shape(), y_shape_original)?;

    let y = ctx.get_output(0)?;
    let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape_original)?;
    y.copy_shape_from_slice(shape);

    Ok(())
}
