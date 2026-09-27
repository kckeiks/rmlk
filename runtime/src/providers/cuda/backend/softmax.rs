use crate::core::allocators::ScratchAllocator;
use crate::core::error::{ConversionError, UnsupportedDataType};
use crate::core::Context;

use crate::providers::cuda::backend::common;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::{attributes, utils};
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct SoftmaxBackend {
    stream: Arc<CudaStream>,
}

impl SoftmaxBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }
}

impl SoftmaxBackend {
    fn compute_softmax<T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let axis = ctx
            .get_attributes()
            .map(|attrs| attributes::softmax::get_axis(&attrs))
            .unwrap_or(-1);

        debug!("[attributes][axis={}]", axis);

        let rank = ctx.get_input(0)?.shape().len();

        // We only support these two axis options.
        if !(axis == -1 || (axis == 1 && rank == 4)) {
            return Err(SoftmaxError::UnsupportedInputValues {
                message: format!("unsupported inputs axis `{axis}` and rank `{rank}"),
            }
            .into());
        }

        // This initializes the output shape.
        common::unary_op_copy_shape(ctx)?;

        let input_tensor = ctx.get_input(0)?;

        debug!(
            "[input][dtype={:?}][shape={:?}][stride=[{:?}]",
            input_tensor.dtype(),
            input_tensor.shape(),
            input_tensor.stride()
        );

        let input_payload = input_tensor.payload();
        let input_data = input_payload.data::<T>();

        let scratch_alloc = ctx.execution_state().scratch_alloc();

        let (input_shape, input_stride) = compute_shape_and_stride(ctx, scratch_alloc, axis)?;

        debug!(
            "[input][processed][shape={:?}][stride=[{:?}]",
            input_shape, input_stride
        );

        let output_tensor = ctx.get_output(0)?;
        output_tensor.init_payload::<T>()?;

        debug!(
            "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
            output_tensor.dtype(),
            output_tensor.shape(),
            output_tensor.stride()
        );

        {
            let mut output_payload = output_tensor.payload_mut();
            let mut output_data = output_payload.data_mut();

            rmlk_cuda::kernels::softmax::compute::<T>(
                &self.stream,
                (T::one(), T::zero()),
                &input_data,
                input_shape,
                input_stride,
                &mut output_data,
                cudarc::cudnn::sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL,
                cudarc::cudnn::sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
            )?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_softmax::<T, T>("debugging/softmax", self.stream.clone(), ctx)
            .unwrap();

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        // Todo: we need to add validation to make sure the tensor types meets
        // the expected data type.
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_softmax::<f32>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn compute_shape_and_stride<'a>(
    ctx: &'a Context<Cuda>,
    scratch_alloc: &'a ScratchAllocator,
    axis: i32,
) -> Result<(&'a [i32], &'a [i32])> {
    let input_tensor = ctx.get_input(0)?;

    // Notice that the tensor will never be updated with this shape.
    // These are only needed for the duration of this computation and then thrown away.
    match axis == -1 {
        true => {
            let shape = scratch_alloc.allocate_fill(4usize, 1i32)?;
            let stride = scratch_alloc.allocate(4usize)?;
            flatten_to_softmax_channel_shape(&input_tensor.shape(), shape)?;
            utils::compute_stride(shape, stride);
            Ok((shape, stride))
        }
        false => {
            let shape = scratch_alloc.allocate_and_convert_from_slice(&input_tensor.shape())?;
            let stride = scratch_alloc.allocate_and_convert_from_slice(&input_tensor.stride())?;
            Ok((shape, stride))
        }
    }
}

/// This function flattens all dimensions except the last one into a single batch dimension,
/// and uses the last dimension as the "channel" dimension (which is where cuDNN will apply softmax).
/// The resulting shape is `[N, C, ...]`. Notice that this function only updates the first two dimensions.
fn flatten_to_softmax_channel_shape<T>(shape: &[usize], dst: &mut [T]) -> Result<()>
where
    T: TryFrom<usize>,
{
    let rank = shape.len();
    let batch_dim = shape[..rank - 1].iter().product::<usize>();
    dst[0] = T::try_from(batch_dim).map_err(|_| ConversionError)?;
    dst[1] = T::try_from(shape[rank - 1]).map_err(|_| ConversionError)?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum SoftmaxError {
    #[error("unsupported input values: {message}")]
    UnsupportedInputValues { message: String },
}
