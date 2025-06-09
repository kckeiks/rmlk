use crate::attributes::conv::ConvAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::conv::BiasInput;
use rmlk_schema::{DataType, DataTypeMap, Op};
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
            _ => {
                unreachable!("we already checked the dimensions of x for the supported dimensions")
            }
        };

        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)?;

        let conv_attrs =
            ConvAttributes::new(&attrs, ctx.execution_state().scratch_alloc(), filter_dims)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let mut y_shape = scratch_alloc.allocate_fill(x.shape().len(), 0)?;

        let w = ctx.get_input(1)?;
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

    fn compute_convolution<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: ConvolutionKernel,
    {
        // We compute the output shape first.
        // This is cheap because we're using scratch buffers.
        self.compute_output_shape(ctx)?;

        let x = ctx.get_input(0)?;

        let filter_dims = match x.shape().len() {
            4 => 2,
            5 => 3,
            _ => {
                unreachable!("we already checked the dimensions of x for the supported dimensions")
            }
        };

        let attrs = ctx
            .get_attributes()
            .ok_or(InternalError::MissingAttributes)?;

        let conv_attrs =
            ConvAttributes::new(&attrs, ctx.execution_state().scratch_alloc(), filter_dims)?;

        let w = ctx.get_input(1)?;
        let bias = ctx.get_input(2).ok();
        let y = ctx.get_output(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;
        let w_shape = scratch_alloc.allocate_and_convert_from_slice(&w.shape())?;
        let y_shape = scratch_alloc.allocate_and_convert_from_slice(y.shape())?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(y.stride())?;

        debug!("[x][conv][shape={:?}][stride=[{:?}]", x.shape(), x.stride());
        debug!("[w][conv][shape={:?}][stride=[{:?}]", w.shape(), w.stride());
        debug!("[y][conv][shape={:?}][stride=[{:?}]", y.shape(), y.stride());

        // Todo: refactor this.
        // Extract and prepare bias argument.
        // At this point, we still don't know the data type of bias.
        let bias = match bias.as_ref() {
            Some(bias_tensor) => {
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

        let elem_count = y_shape.iter().map(|d| *d as usize).product();

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
                    .alloc_zeros::<f32>(elem_count)
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

        // Since we know the data type, we extract it.
        match bias.as_ref() {
            Some(bias) => {
                let data = bias.data.data();
                T::execute::<D>(
                    self.stream,
                    D::one(),
                    D::zero(),
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
                T::execute::<D>(
                    self.stream,
                    D::one(),
                    D::zero(),
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

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: ConvolutionKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_convolution::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Conv,
                dtype,
            }),
        }
    }
}

struct BiasArg<'a, T> {
    data: T,
    shape: &'a [i32],
    stride: &'a [i32],
}

pub trait ConvolutionKernel {
    fn execute<T>(
        stream: Arc<CudaStream>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        w_data: &CudaSlice<T>,
        w_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        dilations: &[i32],
        group: i32,
        bias: Option<BiasInput<T>>,
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl ConvolutionKernel for ActiveKernel {
    fn execute<T>(
        stream: Arc<CudaStream>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        w_data: &CudaSlice<T>,
        w_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        dilations: &[i32],
        group: i32,
        bias: Option<BiasInput<T>>,
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::conv::compute::<T>(
            stream,
            (alpha, beta),
            x_data,
            x_shape,
            x_stride,
            w_data,
            w_shape,
            pads,
            strides,
            dilations,
            group,
            bias,
            y_data,
            y_shape,
            y_stride,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl ConvolutionKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaStream>,
        _: T,
        _: T,
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: i32,
        _: Option<BiasInput<T>>,
        _: &mut CudaSlice<T>,
        _: &[i32],
        _: &[i32],
    ) -> Result<()> {
        Ok(())
    }
}
