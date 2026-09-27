use crate::core::allocators::ScratchAllocator;
use crate::core::error::{ConversionError, UnsupportedDataType};
use crate::core::Context;

use crate::providers::cuda::backend::common;
#[cfg(feature = "dump")]
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
    fn launch_kernel<I, K>(&self, ctx: &Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: ActivationKernel,
    {
        let x_tensor = ctx.get_input(0)?;
        let y_tensor = ctx.get_output(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let (x_shape, x_stride) = alloc_shape_and_stride(scratch_alloc, ctx)?;

        let x_payload = x_tensor.payload();
        let x_data = x_payload.data::<I>();

        let mut y_payload = y_tensor.payload_mut();
        let mut y_data = y_payload.data_mut::<I>();

        K::execute::<I>(
            &self.stream,
            I::one(),
            I::zero(),
            &x_data,
            x_shape,
            x_stride,
            &mut y_data,
        )
    }

    fn compute_activation<I, K>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: ActivationKernel,
    {
        common::unary_op_copy_shape(ctx)?;

        let x = ctx.get_input(0)?;
        debug!(
            "[x][dtype={:?}][shape={:?}][stride=[{:?}]",
            x.dtype(),
            x.shape(),
            x.stride()
        );

        let y_tensor = ctx.get_output(0)?;
        y_tensor.init_payload::<I>()?;

        debug!(
            "[y][dtype={:?}][shape={:?}][stride=[{:?}]",
            I::data_type(),
            y_tensor.shape(),
            y_tensor.stride()
        );

        self.launch_kernel::<I, K>(ctx)?;

        #[cfg(feature = "dump")]
        debug::write_results_unary::<I, I>(
            "debugging/activation",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

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
            _ => Err(UnsupportedDataType(dtype).into()),
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

fn alloc_shape_and_stride<'a>(
    scratch_alloc: &'a ScratchAllocator,
    ctx: &Context<Cuda>,
) -> Result<(&'a [i32], &'a [i32])> {
    let x = ctx.get_input(0)?;

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
            shape_tmp[start + i] = x.shape()[i].to_i32().ok_or(ConversionError)?;
        }
        utils::compute_stride(shape_tmp, stride_tmp);
        (shape_tmp, stride_tmp)
    } else {
        (
            scratch_alloc.allocate_and_convert_from_slice(&x.shape())?,
            scratch_alloc.allocate_and_convert_from_slice(&x.stride())?,
        )
    };

    Ok((x_shape, x_stride))
}
