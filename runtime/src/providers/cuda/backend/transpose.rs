use crate::attributes::transpose;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use rmlk_cuda::kernels::tranpose;
use rmlk_cuda::kernels::tranpose::TransposeKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct TransposeBackend {
    stream: Arc<CudaStream>,
}

impl TransposeBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }

    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => TransposeKernel::FwdF16,
            DataType::Float => TransposeKernel::FwdF32,
            DataType::Double => TransposeKernel::FwdF64,
            DataType::Int32 => TransposeKernel::FwdI32,
            DataType::Int64 => TransposeKernel::FwdI64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        tranpose::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_transpose<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Default + Clone + Copy,
    {
        {
            let input = ctx.get_input(0)?;

            debug!(
                "[input][dtype={:?}][shape={:?}][stride=[{:?}]",
                input.dtype(),
                input.shape(),
                input.stride()
            );

            let alloc = ctx.execution_state().scratch_alloc().clone();

            let output_shape = alloc.allocate(input.shape().len())?;

            let perm = match ctx
                .get_attributes()
                .as_ref()
                .and_then(|attrs| transpose::get_perm(&attrs))
            {
                None => {
                    let perm = alloc.allocate(input.shape().len())?;
                    for (dst_i, (src_i, dim)) in input.shape().iter().enumerate().rev().enumerate()
                    {
                        output_shape[dst_i] = *dim;
                        perm[dst_i] = src_i;
                    }
                    debug!("[perm={:?}]", perm);
                    perm
                }
                Some(perm) => {
                    debug!("[perm={:?}]", perm);

                    // The length of perm must be equal to the rank of the input.
                    if perm.len() != output_shape.len() {
                        return Err(TransposeError::InvalidPermLength.into());
                    }

                    for (dst_i, dim_i) in perm.iter().enumerate() {
                        // Todo: add a more detailed error message.
                        let i = usize::try_from(*dim_i)
                            .map_err(|_| InternalError::UnableToConvertValue)?;
                        if i >= output_shape.len() {
                            return Err(
                                TransposeError::PermIndexOutOfBounds { index: *dim_i }.into()
                            );
                        }
                        output_shape[dst_i] = input.shape()[i];
                    }

                    alloc.allocate_and_convert_from_slice(perm)?
                }
            };

            let perm_on_dev = self
                .stream
                .memcpy_stod(perm)
                .map_err(|e| InternalError::Device { error: e.into() })?;

            let dst = ctx.get_output(0)?.dst_id();

            ctx.execution_state_mut()
                .copy_shape_from_slice(output_shape, dst)?;

            // Todo: to avoid problems with non-contiguous memory
            // for now we simply clone the data. We can do better.
            let input = ctx.get_input(0)?;
            let input_dev_data_ptr = input.try_dev_data_ptr()?;
            let input_view = input_dev_data_ptr.data::<I>();

            common::init_tensor_device_data::<I>(&self.stream, ctx.get_output(0)?)?;

            let output = ctx.get_output(0)?;

            debug!(
                "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
                output.dtype(),
                output.shape(),
                output.stride()
            );

            let func = self.load_cuda_function(I::data_type())?;

            let rank = output.shape().len();

            let info = alloc.allocate(3 * rank)?;
            info[..rank].copy_from_slice(output.shape());
            info[rank..2 * rank].copy_from_slice(input.stride());
            info[2 * rank..].copy_from_slice(output.stride());

            let mut output_dev_data_ptr = output.try_dev_data_ptr_mut()?;
            let mut output_view = output_dev_data_ptr.data_mut::<I>();

            unsafe {
                tranpose::compute(
                    self.stream.clone(),
                    func,
                    rank,
                    info,
                    &perm_on_dev,
                    &input_view,
                    &mut output_view,
                )?;
            }
        }

        common::write_results_transpose::<I>("debugging/transpose", self.stream.clone(), ctx)
            .unwrap();

        /*self.stream
            .synchronize()
            .map_err(|e| InternalError::Device { error: e.into() })?;*/

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_transpose::<f16>(ctx),
            DataType::Float => self.compute_transpose::<f32>(ctx),
            DataType::Double => self.compute_transpose::<f64>(ctx),
            DataType::Int32 => self.compute_transpose::<i32>(ctx),
            DataType::Int64 => self.compute_transpose::<i64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

#[derive(Debug)]
pub enum TransposeError {
    InvalidPermLength,
    PermIndexOutOfBounds { index: i32 },
}

impl Display for TransposeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TransposeError::InvalidPermLength => {
                write!(f, "invalid perm length")
            }
            TransposeError::PermIndexOutOfBounds { index } => {
                write!(f, "perm index `{}` out of bounds", index)
            }
        }
    }
}

impl std::error::Error for TransposeError {}
