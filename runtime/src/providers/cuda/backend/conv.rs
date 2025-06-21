use crate::attributes::conv::ConvAttributes;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::conv::BiasInput;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct ConvolutionBackend {
    stream: Arc<CudaStream>,
}

impl ConvolutionBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl ConvolutionBackend {
    fn compute_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let filter_dims = match x.shape().len() {
            4 => 2,
            5 => 3,
            rank => return Err(ConvError::InvalidInputRank { rank }.into()),
        };

        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)
            .map_err(Box::new)?;

        let conv_attrs =
            ConvAttributes::new(&attrs, ctx.execution_state().scratch_alloc(), filter_dims)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let mut y_shape = scratch_alloc.allocate_fill(x.shape().len(), 0)?;

        let w = ctx.get_input(1)?;

        if w.is_scalar() {
            return Err(ConvError::InvalidInputRank { rank: 0 }.into());
        }

        let w_shape = scratch_alloc.allocate_and_convert_from_slice(&w.shape())?;

        // Todo: update this function so we dont have to do all this work with
        // allocating scratch buffers.
        rmlk_cuda::kernels::conv::calculate_output_shape(
            &x_shape,
            &w_shape,
            conv_attrs.pads(),
            conv_attrs.strides(),
            conv_attrs.dilations(),
            &mut y_shape,
        )?;

        let y = ctx.get_output(0)?;
        let y_index = y.dst_id();
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        ctx.execution_state_mut()
            .copy_shape_from_slice(shape, y_index)?;

        Ok(())
    }

    fn compute_convolution<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        // We compute the output shape first.
        // This is cheap because we're using scratch buffers.
        self.compute_output_shape(ctx)?;

        let x = ctx.get_input(0)?;

        debug!(
            "[x][dtype={:?}][shape={:?}][stride=[{:?}]",
            x.dtype(),
            x.shape(),
            x.stride()
        );

        let filter_dims = match x.shape().len() {
            4 => 2,
            5 => 3,
            _ => {
                unreachable!("we already checked the dimensions of x for the supported dimensions")
            }
        };

        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)
            .map_err(Box::new)?;

        let conv_attrs =
            ConvAttributes::new(&attrs, ctx.execution_state().scratch_alloc(), filter_dims)?;

        let w = ctx.get_input(1)?;

        debug!(
            "[w][dtype={:?}][shape={:?}][stride=[{:?}]",
            w.dtype(),
            w.shape(),
            w.stride()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;
        let w_shape = scratch_alloc.allocate_and_convert_from_slice(&w.shape())?;

        let y = ctx.get_output(0)?;
        let y_shape = scratch_alloc.allocate_and_convert_from_slice(y.shape())?;

        let y_stride = scratch_alloc.allocate_and_convert_from_slice(y.stride())?;

        // Todo: refactor this.
        // Extract and prepare bias argument.
        // At this point, we still don't know the data type of bias.
        let bias = ctx.get_input(2).ok();
        let bias = match bias.as_ref() {
            Some(bias_tensor) => {
                debug!(
                    "[bias][dtype={:?}][shape={:?}][stride=[{:?}]",
                    bias_tensor.dtype(),
                    bias_tensor.shape(),
                    bias_tensor.stride()
                );

                if bias_tensor.is_scalar() {
                    return Err(ConvError::InvalidInputRank { rank: 0 }.into());
                }

                let bias_shape = scratch_alloc.allocate_fill(x_shape.len(), 1)?;
                // Todo: Urgent. We need to make this generic.
                bias_shape[1] = bias_tensor.shape()[0] as i32;

                let bias_stride = scratch_alloc.allocate_fill(x_shape.len(), 0)?;
                utils::compute_stride(&bias_shape, bias_stride);

                let device_data = bias_tensor.try_dev_data_ptr()?;

                Some(BiasArg {
                    data: device_data,
                    shape: bias_shape,
                    stride: bias_stride,
                })
            }
            None => None,
        };

        let x_dev_data_ref = x.try_dev_data_ptr()?;
        let x_dev_data = x_dev_data_ref.data();

        let w_dev_data_ref = w.try_dev_data_ptr()?;
        let w_dev_data = w_dev_data_ref.data();

        let y = ctx.get_output(0)?;

        debug!(
            "[y][dtype={:?}][shape={:?}][stride=[{:?}]",
            D::data_type(),
            y.shape(),
            y.stride()
        );

        common::init_tensor_device_data::<D>(&self.stream, y)?;

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        // Since we know the data type, we extract it.
        match bias.as_ref() {
            Some(bias) => {
                let data = bias.data.data();
                rmlk_cuda::kernels::conv::compute::<D>(
                    self.stream,
                    (D::one(), D::zero()),
                    &x_dev_data,
                    &x_shape,
                    &x_stride,
                    &w_dev_data,
                    &w_shape,
                    conv_attrs.pads(),
                    conv_attrs.strides(),
                    conv_attrs.dilations(),
                    conv_attrs.group(),
                    Some(BiasInput {
                        data: &data,
                        shape: bias.shape,
                        stride: bias.stride,
                    }),
                    &mut y_dev_data,
                    &y_shape,
                    &y_stride,
                )?;
            }
            None => {
                rmlk_cuda::kernels::conv::compute::<D>(
                    self.stream,
                    (D::one(), D::zero()),
                    &x_dev_data,
                    &x_shape,
                    &x_stride,
                    &w_dev_data,
                    &w_shape,
                    conv_attrs.pads(),
                    conv_attrs.strides(),
                    conv_attrs.dilations(),
                    conv_attrs.group(),
                    None,
                    &mut y_dev_data,
                    &y_shape,
                    &y_stride,
                )?;
            }
        };

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_convolution::<f16>(ctx),
            DataType::Float => self.compute_convolution::<f32>(ctx),
            DataType::Double => self.compute_convolution::<f64>(ctx),
            DataType::Int32 => self.compute_convolution::<i32>(ctx),
            DataType::Int64 => self.compute_convolution::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

struct BiasArg<'a, T> {
    data: T,
    shape: &'a [i32],
    stride: &'a [i32],
}

#[derive(Debug)]
pub enum ConvError {
    InvalidInputRank { rank: usize },
}

impl Display for ConvError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvError::InvalidInputRank { rank } => {
                write!(f, "Invalid input rank: {}", rank)
            }
        }
    }
}

impl std::error::Error for ConvError {}
