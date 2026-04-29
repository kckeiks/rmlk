use crate::attributes::reduce_mean;
use crate::core::error::UnsupportedDataType;
use crate::core::Context;

use crate::providers::cuda;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::reduce_mean::ReduceKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct ReduceMeanBackend {
    stream: Arc<CudaStream>,
}

impl ReduceMeanBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }

    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel = match dtype {
            DataType::Float16 => ReduceKernel::FwdF16,
            DataType::Float => ReduceKernel::FwdF32,
            DataType::Double => ReduceKernel::FwdF64,
            DataType::Int32 => ReduceKernel::FwdI32,
            DataType::Int64 => ReduceKernel::FwdI64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel);

        Ok(rmlk_cuda::kernels::reduce_mean::load_kernel(
            self.stream.context().clone(),
            kernel,
        )?)
    }

    fn compute_reduce_mean<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(T::data_type())?;

        let input = ctx.get_input(0)?;

        debug!(
            "[input][dtype={:?}][shape={:?}][stride={:?}]",
            input.dtype(),
            input.shape(),
            input.stride()
        );

        // Todo: document or fix.
        // In the compute_output_shape helper, we don't do this.
        let rank = if input.is_scalar() {
            1
        } else {
            input.shape().len()
        };

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let attrs = ctx.get_attributes();
        let noop_with_empty_axes = attrs
            .as_ref()
            .map(|attrs| reduce_mean::get_noop_with_empty_axes(&attrs))
            .unwrap_or(false);

        // We either return a non-empty axes list or nothing.
        let axes = {
            match ctx.get_input(1).ok() {
                Some(tensor) => {
                    debug!(
                        "[axes][dtype={:?}][shape={:?}][stride={:?}]",
                        tensor.dtype(),
                        tensor.shape(),
                        tensor.stride()
                    );

                    if !tensor.is_empty() {
                        let axes_data_len = tensor.len();

                        let raw_axes = alloc.allocate::<i64>(axes_data_len)?;
                        tensor.payload_to_host(raw_axes)?;

                        let axes = alloc.allocate::<usize>(axes_data_len)?;
                        utils::normalize_indices(raw_axes, axes, rank)?;

                        Some(axes)
                    } else if !tensor.is_scalar() {
                        None
                    } else {
                        return Err(ReduceMeanError::ScalarAxisNotAllowed.into());
                    }
                }
                None => {
                    if let Some(raw_axes) =
                        attrs.as_ref().and_then(|attr| reduce_mean::get_axes(&attr))
                    {
                        if !raw_axes.is_empty() {
                            let raw_axes =
                                alloc.allocate_and_convert_from_slice::<i32, i64>(raw_axes)?;
                            let axes = alloc.allocate::<usize>(raw_axes.len())?;
                            utils::normalize_indices(raw_axes, axes, rank)?;
                            Some(axes)
                        } else {
                            None
                        }
                    } else {
                        let buf = alloc.allocate::<usize>(rank)?;
                        utils::write_increasing_sequence(buf)?;

                        debug!("[No `AXES`][axes={buf:?}]");

                        Some(buf)
                    }
                }
            }
        };

        if (axes.is_none() && !noop_with_empty_axes) || axes.is_some() {
            let axes = match axes {
                None => {
                    let buf = alloc.allocate::<usize>(rank)?;
                    utils::write_increasing_sequence(buf)?;
                    buf
                }
                Some(axes) => axes,
            };

            compute_output_shape(&axes, ctx)?;

            let output = ctx.get_output(0)?;

            let keep_dims = ctx
                .get_attributes()
                .map(|attrs| reduce_mean::get_keep_dims(&attrs))
                .unwrap_or(true);

            // The output is a scalar if and only if:
            // 1. The input is a scalar.
            // 2. All axes are reduced and keepdims = false.
            if input.is_scalar() || output.shape().is_empty() && !keep_dims {
                output.init_scalar_payload::<T>()?;
            } else {
                output.init_payload::<T>()?;
            }

            let input = ctx.get_input(0)?;

            let (input_shape, input_stride) = if input.is_scalar() {
                cuda::utils::scalar_shape_and_stride(&input)
            } else {
                (input.shape(), input.stride())
            };

            let mut reduced_dim_prod = 1;
            for axis in axes.iter().copied() {
                reduced_dim_prod *= input_shape[axis]
            }

            let info_on_host = alloc.allocate(2 * rank)?;
            info_on_host[..rank].copy_from_slice(&input_shape);
            info_on_host[rank..2 * rank].copy_from_slice(&input_stride);

            let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
            let info = cuda_bump.alloc_from_slice_with_fallback(info_on_host)?;
            let info_data = info.data::<usize>();

            let axes = cuda_bump.alloc_from_slice_with_fallback(axes)?;
            let axes_data = axes.data::<usize>();

            let input_payload = input.payload();
            let input_data = input_payload.data::<T>();

            debug!(
                "[output][dtype={:?}][shape={:?}][stride={:?}]",
                output.dtype(),
                output.shape(),
                output.stride()
            );

            let mut output_payload = output.payload_mut();
            let mut output_data = output_payload.data_mut::<T>();

            unsafe {
                rmlk_cuda::kernels::reduce_mean::compute(
                    self.stream.clone(),
                    func,
                    reduced_dim_prod,
                    &axes_data,
                    rank,
                    &info_data,
                    &input_data,
                    &mut output_data,
                )?;
            }
        } else {
            copy_input_to_output::<T>(ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_reduce_mean::<T, i64>(
            "debugging/reduce_mean",
            self.stream.clone(),
            ctx,
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_reduce_mean::<f16>(ctx),
            DataType::Float => self.compute_reduce_mean::<f32>(ctx),
            DataType::Double => self.compute_reduce_mean::<f64>(ctx),
            DataType::Int32 => self.compute_reduce_mean::<i32>(ctx),
            DataType::Int64 => self.compute_reduce_mean::<i64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn copy_input_to_output<I>(ctx: &mut Context<Cuda>) -> Result<()>
where
    I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
{
    let input = ctx.get_input(0)?;
    let output = ctx.get_output(0)?;
    output.copy_shape(input.shape_handle());

    let input_payload = input.payload();
    let input_data = input_payload.data::<I>();

    output.write_payload(&input_data)?;

    Ok(())
}

fn compute_output_shape(axes: &[usize], ctx: &Context<Cuda>) -> Result<()> {
    let input = ctx.get_input(0)?;
    let rank = input.shape().len();

    let alloc = ctx.execution_state().scratch_alloc().clone();
    let reduced = alloc.allocate::<bool>(rank)?;

    // Todo: this if was added to handle scalars. Revisit.
    if rank > 0 {
        for axis in axes.iter().copied() {
            if axis > rank {
                return Err(ReduceMeanError::AxisOutOfBounds.into());
            }

            if reduced[axis] {
                return Err(ReduceMeanError::DuplicateAxis.into());
            }

            reduced[axis] = true;
        }
    }

    let keep_dims = ctx
        .get_attributes()
        .map(|attrs| reduce_mean::get_keep_dims(&attrs))
        .unwrap_or(true);

    let output_shape = if keep_dims {
        let buf = alloc.allocate_from_slice(&input.shape())?;
        for dim in 0..rank {
            if reduced[dim] {
                buf[dim] = 1;
            }
        }
        buf
    } else {
        if rank < axes.len() {
            return Err(ReduceMeanError::AxesLargerThanRank.into());
        }

        let buf = alloc.allocate::<usize>(rank - axes.len())?;
        for (axis, dim) in input
            .shape()
            .iter()
            .copied()
            .enumerate()
            .filter(|(axis, _dim)| !reduced[*axis])
            .map(|(_, dim)| dim)
            .enumerate()
        {
            buf[axis] = dim;
        }
        buf
    };

    let output_tensor = ctx.get_output(0)?;
    output_tensor.copy_shape_from_slice(&output_shape);

    Ok(())
}

#[derive(Debug)]
pub enum ReduceMeanError {
    AxisOutOfBounds,
    DuplicateAxis,
    AxesLargerThanRank,
    ScalarAxisNotAllowed,
}

impl Display for ReduceMeanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ReduceMeanError {}
