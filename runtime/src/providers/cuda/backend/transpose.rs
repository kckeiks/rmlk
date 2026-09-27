use crate::core::allocators::ScratchAllocator;
use crate::core::error::{ConversionError, UnsupportedDataType};
use crate::core::Context;

use crate::attributes;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use rmlk_cuda::kernels::transpose;
use rmlk_cuda::kernels::transpose::TransposeKernel;
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
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        transpose::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_transpose<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Default + Clone + Copy,
    {
        let input = ctx.get_input(0)?;

        debug!(
            "[input][dtype={:?}][shape={:?}][stride=[{:?}]",
            input.dtype(),
            input.shape(),
            input.stride()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let (perm, output_shape) = compute_perm_and_output_shape(ctx, &scratch_alloc)?;

        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let perm_on_dev = cuda_bump.alloc_from_slice_with_fallback(perm)?;
        let perm_data = perm_on_dev.data::<usize>();

        let input = ctx.get_input(0)?;
        let input_payload = input.payload();
        let input_data = input_payload.data::<T>();

        let output = ctx.get_output(0)?;
        output.copy_shape_from_slice(output_shape);

        if input.is_scalar() {
            output.init_scalar_payload::<T>()?;
        } else {
            output.init_payload::<T>()?;
        }

        debug!(
            "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
            output.dtype(),
            output.shape(),
            output.stride()
        );

        let func = self.load_cuda_function(T::data_type())?;

        let rank = output.shape().len();

        let info_on_host = scratch_alloc.allocate(3 * rank)?;
        info_on_host[..rank].copy_from_slice(&output.shape());
        info_on_host[rank..2 * rank].copy_from_slice(&input.stride());
        info_on_host[2 * rank..].copy_from_slice(&output.stride());

        let info = cuda_bump.alloc_from_slice_with_fallback(info_on_host)?;
        let info_data = info.data::<usize>();

        {
            let mut output_payload = output.payload_mut();
            let mut output_data = output_payload.data_mut::<T>();

            unsafe {
                transpose::compute(
                    self.stream.clone(),
                    func,
                    rank,
                    &info_data,
                    &perm_data,
                    &input_data,
                    &mut output_data,
                )?;
            }
        }

        #[cfg(feature = "dump")]
        debug::write_results_transpose::<T>("debugging/transpose", self.stream.clone(), ctx)
            .unwrap();

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
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn compute_perm_and_output_shape<'a>(
    ctx: &Context<Cuda>,
    scratch_alloc: &'a ScratchAllocator,
) -> Result<(&'a [usize], &'a [usize])> {
    let input = ctx.get_input(0)?;
    let output_shape = scratch_alloc.allocate(input.shape().len())?;

    match ctx
        .get_attributes()
        .as_ref()
        .and_then(|attrs| attributes::transpose::get_perm(attrs))
    {
        None => {
            let perm = scratch_alloc.allocate(input.shape().len())?;
            for (dst_i, (src_i, dim)) in input.shape().iter().enumerate().rev().enumerate() {
                output_shape[dst_i] = *dim;
                perm[dst_i] = src_i;
            }
            debug!("[perm={:?}]", perm);
            Ok((perm, output_shape))
        }
        Some(perm) => {
            debug!("[perm={:?}]", perm);

            // The length of perm must be equal to the rank of the input.
            if perm.len() != output_shape.len() {
                return Err(TransposeError::InvalidPermLength.into());
            }

            for (dst_i, dim_i) in perm.iter().enumerate() {
                // Todo: add a more detailed error message.
                let i = usize::try_from(*dim_i).map_err(|_| ConversionError)?;
                if i >= output_shape.len() {
                    return Err(TransposeError::PermIndexOutOfBounds { index: *dim_i }.into());
                }
                output_shape[dst_i] = input.shape()[i];
            }

            Ok((
                scratch_alloc.allocate_and_convert_from_slice(perm)?,
                output_shape,
            ))
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
