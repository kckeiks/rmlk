use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
#[cfg(feature = "debugger")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::{Num, ToPrimitive};
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct ActivationBackend {
    stream: Arc<CudaStream>,
}

impl ActivationBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl ActivationBackend {
    fn compute_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;
        let x_index = x.src_id();
        let y_index = y.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_within(x_index, y_index)
            .map_err(Into::into)
    }

    fn compute_activation<I, K>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: ActivationKernel,
    {
        {
            self.compute_output_shape(ctx)?;

            let x = ctx.get_input(0)?;
            debug!(
                "[x][dtype={:?}][shape={:?}][stride=[{:?}]",
                I::data_type(),
                x.shape(),
                x.stride()
            );

            let scratch_alloc = ctx.execution_state().scratch_alloc();
            let (x_shape, x_stride) = if x.is_scalar() {
                (
                    scratch_alloc.allocate_from_slice(&[1])?,
                    scratch_alloc.allocate_from_slice(&[1])?,
                )
            } else if x.shape().len() < 4 {
                let shape_tmp = scratch_alloc.allocate_fill::<i32>(4, 1)?;
                let stride_tmp = scratch_alloc.allocate_fill::<i32>(4, 0)?;
                let start = 4 - x.shape().len();
                for i in 0..x.shape().len() {
                    shape_tmp[start + i] = x.shape()[i]
                        .to_i32()
                        .ok_or(InternalError::UnableToConvertValue)?;
                }
                utils::compute_stride(shape_tmp, stride_tmp);
                (shape_tmp, stride_tmp)
            } else {
                (
                    scratch_alloc.allocate_and_convert_from_slice(&x.shape())?,
                    scratch_alloc.allocate_and_convert_from_slice(&x.stride())?,
                )
            };

            let x_dev_data_ref = x.try_dev_data_ptr()?;
            let x_dev_data = x_dev_data_ref.data::<I>();

            let y_tensor = ctx.get_output(0)?;

            debug!(
                "[y][dtype={:?}][shape={:?}][stride=[{:?}]",
                I::data_type(),
                y_tensor.shape(),
                y_tensor.stride()
            );

            common::init_tensor_device_data::<I>(&self.stream, y_tensor)?;

            let y_tensor = ctx.get_output(0)?;
            let mut y_ptr = y_tensor.dev_data_ptr_mut();
            let mut y_view = y_ptr
                .as_mut()
                .expect("we already checked that it initialized")
                .data_mut();

            K::execute::<I>(
                &self.stream,
                I::one(),
                I::zero(),
                &x_dev_data,
                x_shape,
                x_stride,
                &mut y_view,
            )?;
        }

        #[cfg(feature = "debugger")]
        debug::write_results_unary::<I, I>(
            "debugging/activation",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        /*self.stream
        .synchronize()
        .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: ActivationKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_activation::<f16, T>(ctx),
            DataType::Float => self.compute_activation::<f32, T>(ctx),
            DataType::Double => self.compute_activation::<f64, T>(ctx),
            DataType::Int32 => self.compute_activation::<i32, T>(ctx),
            DataType::Int64 => self.compute_activation::<i64, T>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

pub trait ActivationKernel {
    fn execute<T>(
        stream: &Arc<CudaStream>,
        alpha: T,
        beta: T,
        x_dev_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}
