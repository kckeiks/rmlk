use crate::attributes::conv::ConvAttributes;
use crate::attributes::error::AttributeError;
use crate::core::allocators::ScratchAllocator;
use crate::core::error::UnsupportedDataType;
use crate::core::Context;
use crate::providers::cuda::{Cuda, Tensor};
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
    fn compute_output_shape(&self, ctx: &Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let filter_dims = match x.shape().len() {
            4 => 2,
            5 => 3,
            rank => return Err(ConvError::InvalidInputRank { rank }.into()),
        };

        let attrs = ctx
            .get_attributes()
            .ok_or(AttributeError::MissingAttributes)
            .map_err(Box::new)?;

        let conv_attrs =
            ConvAttributes::new(&attrs, ctx.execution_state().scratch_alloc(), filter_dims)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let y_shape = scratch_alloc.allocate_fill(x.shape().len(), 0)?;

        let w = ctx.get_input(1)?;

        if w.is_scalar() {
            return Err(ConvError::InvalidInputRank { rank: 0 }.into());
        }

        let w_shape = scratch_alloc.allocate_and_convert_from_slice(&w.shape())?;

        // Todo: update this function so we dont have to do all this work with
        // allocating scratch buffers.
        rmlk_cuda::kernels::conv::calculate_output_shape(
            x_shape,
            w_shape,
            conv_attrs.pads(),
            conv_attrs.strides(),
            conv_attrs.dilations(),
            y_shape,
        )?;

        let y = ctx.get_output(0)?;
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        y.copy_shape_from_slice(shape);

        Ok(())
    }

    fn compute_convolution<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
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
            .ok_or(AttributeError::MissingAttributes)
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
        let y_shape = scratch_alloc.allocate_and_convert_from_slice(&y.shape())?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(&y.stride())?;

        let bias = get_bias(ctx, &scratch_alloc)?;

        let x_payload = x.payload();
        let x_data = x_payload.data();

        let w_payload = w.payload();
        let w_data = w_payload.data();

        let y = ctx.get_output(0)?;
        y.init_payload::<T>()?;

        debug!(
            "[y][dtype={:?}][shape={:?}][stride=[{:?}]",
            T::data_type(),
            y.shape(),
            y.stride()
        );

        let mut y_payload = y.payload_mut();
        let mut y_dev = y_payload.data_mut();

        // Since we know the data type, we extract it.
        match bias.as_ref() {
            Some(bias) => {
                let bias_payload = bias.data.payload();
                let bias_data = bias_payload.data();
                rmlk_cuda::kernels::conv::compute::<T>(
                    self.stream,
                    (T::one(), T::zero()),
                    &x_data,
                    x_shape,
                    x_stride,
                    &w_data,
                    w_shape,
                    conv_attrs.pads(),
                    conv_attrs.strides(),
                    conv_attrs.dilations(),
                    conv_attrs.group(),
                    Some(BiasInput {
                        data: &bias_data,
                        shape: bias.shape,
                        stride: bias.stride,
                    }),
                    &mut y_dev,
                    y_shape,
                    y_stride,
                )?;
            }
            None => {
                rmlk_cuda::kernels::conv::compute::<T>(
                    self.stream,
                    (T::one(), T::zero()),
                    &x_data,
                    x_shape,
                    x_stride,
                    &w_data,
                    w_shape,
                    conv_attrs.pads(),
                    conv_attrs.strides(),
                    conv_attrs.dilations(),
                    conv_attrs.group(),
                    None,
                    &mut y_dev,
                    y_shape,
                    y_stride,
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
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

// Todo: refactor this.
// Extract and prepare bias argument.
// At this point, we still don't know the data type of bias.
fn get_bias<'a>(
    ctx: &'a Context<Cuda>,
    scratch_alloc: &'a ScratchAllocator,
) -> Result<Option<BiasArg<'a, Tensor>>> {
    let x = ctx.get_input(0)?;
    let bias = ctx.get_input(2).ok();
    let bias = match bias {
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

            let bias_shape = scratch_alloc.allocate_fill(x.shape().len(), 1)?;
            // Todo: Urgent. We need to make this generic.
            bias_shape[1] = bias_tensor.shape()[0] as i32;

            let bias_stride = scratch_alloc.allocate_fill(x.shape().len(), 0)?;
            utils::compute_stride(bias_shape, bias_stride);

            Some(BiasArg {
                data: bias_tensor,
                shape: bias_shape,
                stride: bias_stride,
            })
        }
        None => None,
    };

    Ok(bias)
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

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::{AttributeType, Op};

    #[test]
    fn oneshot_scale() {
        // A 1x1 convolution that scales every element by 2.
        // Pads must have length at least 4 because ConvAttributes::pads returns the
        // spatial slice starting at index 2.
        let out = OpTest::new(Op::Conv)
            .input([1, 1, 2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([1, 1, 1, 1], vec![2.0f32])
            .attr("pads", AttributeType::Ints(vec![0, 0, 0, 0]))
            .attr("strides", AttributeType::Ints(vec![1, 1]))
            .attr("dilations", AttributeType::Ints(vec![1, 1]))
            .output([1, 1, 2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn three_by_three_all_ones() {
        let out = OpTest::new(Op::Conv)
            .input([1, 1, 3, 3], vec![1.0f32; 9])
            .input([1, 1, 3, 3], vec![1.0f32; 9])
            .attr("pads", AttributeType::Ints(vec![0, 0, 0, 0]))
            .attr("strides", AttributeType::Ints(vec![1, 1]))
            .attr("dilations", AttributeType::Ints(vec![1, 1]))
            .output([1, 1, 1, 1])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![9.0]);
    }

    #[test]
    fn with_bias() {
        let out = OpTest::new(Op::Conv)
            .input([1, 1, 2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([1, 1, 1, 1], vec![1.0f32])
            .input([1], vec![10.0f32])
            .attr("pads", AttributeType::Ints(vec![0, 0, 0, 0]))
            .attr("strides", AttributeType::Ints(vec![1, 1]))
            .attr("dilations", AttributeType::Ints(vec![1, 1]))
            .output([1, 1, 2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Conv)
            .input([1, 1, 1, 1], vec![true])
            .input([1, 1, 1, 1], vec![true])
            .attr("pads", AttributeType::Ints(vec![0, 0, 0, 0]))
            .output([1, 1, 1, 1])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
