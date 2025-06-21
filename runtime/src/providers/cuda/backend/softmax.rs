use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
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
    fn compute_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let input = ctx.get_input(0)?;
        let output = ctx.get_output(0)?;
        let input_id = input.src_id();
        let output_id = output.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_within(input_id, output_id)
            .map_err(Into::into)
    }

    fn compute_softmax<T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        {
            let axis = ctx
                .get_attributes()
                .map(|attrs| attributes::softmax::get_axis(&attrs))
                .unwrap_or(-1);

            debug!("[attributes][axis={}]", axis);

            let rank = ctx.get_input(0)?.shape().len();

            // We only support these two axis options.
            if !(axis == -1 || (axis == 1 && rank == 4)) {
                return Err(InternalError::UnsupportedInputValues {
                    message: format!("unsupported inputs axis `{axis}` and rank `{rank}"),
                }
                .into());
            }

            // This initializes the output shape.
            self.compute_output_shape(ctx)?;

            let input_tensor = ctx.get_input(0)?;

            debug!(
                "[input][dtype={:?}][shape={:?}][stride=[{:?}]",
                input_tensor.dtype(),
                input_tensor.shape(),
                input_tensor.stride()
            );

            let scratch_alloc = ctx.execution_state().scratch_alloc();

            // Notice that the tensor will never be updated with this shape.
            // These are only needed for the duration of this computation and then thrown away.
            let (input_shape, input_stride) = match axis == -1 {
                true => {
                    let shape = scratch_alloc.allocate_fill(4usize, 1i32)?;
                    let stride = scratch_alloc.allocate(4usize)?;
                    flatten_to_softmax_channel_shape(input_tensor.shape(), shape)?;
                    utils::compute_stride(shape, stride);
                    (shape, stride)
                }
                false => {
                    let shape =
                        scratch_alloc.allocate_and_convert_from_slice(&input_tensor.shape())?;
                    let stride =
                        scratch_alloc.allocate_and_convert_from_slice(&input_tensor.stride())?;
                    (shape, stride)
                }
            };

            debug!(
                "[input][processed][shape={:?}][stride=[{:?}]",
                input_shape, input_stride
            );

            let input_dev_ptr = input_tensor.try_dev_data_ptr()?;
            let input_data_view = input_dev_ptr.data::<T>();

            let output_tensor = ctx.get_output(0)?;

            debug!(
                "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
                output_tensor.dtype(),
                output_tensor.shape(),
                output_tensor.stride()
            );

            common::init_tensor_device_data::<T>(&self.stream, output_tensor)?;

            let output_tensor = ctx.get_output(0)?;
            let mut output_dev_ptr = output_tensor.dev_data_ptr_mut();
            let mut output_data_view = output_dev_ptr
                .as_mut()
                .expect("we already checked that it initialized")
                .data_mut();

            rmlk_cuda::kernels::softmax::compute::<T>(
                &self.stream,
                (T::one(), T::zero()),
                &input_data_view,
                input_shape,
                input_stride,
                &mut output_data_view,
                cudarc::cudnn::sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL,
                cudarc::cudnn::sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
            )?;
        }

        common::write_results_softmax::<T, T>("debugging/softmax", self.stream.clone(), ctx)
            .unwrap();

        /*self.stream
            .synchronize()
            .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        // Todo: we need to add validation to make sure the tensor types meets
        // the expected data type.
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_softmax::<f32>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
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
    dst[0] = T::try_from(batch_dim).map_err(|_| InternalError::UnableToConvertValue)?;
    dst[1] = T::try_from(shape[rank - 1]).map_err(|_| InternalError::UnableToConvertValue)?;
    Ok(())
}
