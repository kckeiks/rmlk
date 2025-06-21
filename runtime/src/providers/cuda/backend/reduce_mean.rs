use crate::attributes::reduce_mean;
use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::data::CudaData;
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
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        debug!("[kernel={:?}]", kernel);

        Ok(rmlk_cuda::kernels::reduce_mean::load_kernel(
            self.stream.context().clone(),
            kernel,
        )?)
    }

    fn compute_output_shape(&mut self, axes: &[usize], ctx: &mut Context<Cuda>) -> Result<()> {
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
            let buf = alloc.allocate_from_slice(input.shape())?;
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

        let dst = ctx.get_output(0)?.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst)?;
        Ok(())
    }

    fn copy_input_to_output<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        {
            let src = ctx.get_input(0)?.src_id();
            let dst = ctx.get_output(0)?.dst_id();
            ctx.execution_state_mut().copy_shape_from_within(src, dst)?;
        }

        let input = ctx.get_input(0)?;
        let input_dev_data_ptr = input.try_dev_data_ptr()?;
        let dev_data = input_dev_data_ptr.data::<I>().clone();

        let mut output = ctx.get_output(0)?;

        output.set_dev_data(CudaData::new(dev_data));

        Ok(())
    }

    fn compute_reduce_mean<I>(mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(I::data_type())?;

        let input = ctx.get_input(0)?;

        debug!(
            "[input][dtype={:?}][shape={:?}][stride={:?}]",
            input.dtype(),
            input.shape(),
            input.stride()
        );

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

                    let axes_dev_ptr = tensor.try_dev_data_ptr()?;
                    let axes_dev_data = axes_dev_ptr.data::<i64>();

                    if !axes_dev_data.is_empty() {
                        let raw_axes = alloc.allocate::<i64>(axes_dev_data.len())?;
                        self.stream
                            .memcpy_dtoh(axes_dev_data.as_ref(), raw_axes)
                            .map_err(rmlk_cuda::Error::from)?;
                        let axes = alloc.allocate::<usize>(axes_dev_data.len())?;
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
                        debug!("[no `AXES` tensor]");
                        let buf = alloc.allocate::<usize>(rank)?;
                        utils::write_increasing_sequence(buf)?;
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

            self.compute_output_shape(&axes, ctx)?;

            let input = ctx.get_input(0)?;
            let (input_shape, input_stride) = if input.is_scalar() {
                ([1].as_ref(), [1].as_ref())
            } else {
                (input.shape(), input.stride())
            };

            let mut reduced_dim_prod = 1;
            for axis in axes.iter().copied() {
                reduced_dim_prod *= input_shape[axis]
            }

            let info = alloc.allocate(2 * rank)?;
            info[..rank].copy_from_slice(input_shape);
            info[rank..2 * rank].copy_from_slice(input_stride);

            let input_dev_ptr = input.try_dev_data_ptr()?;
            let input_dev_data = input_dev_ptr.data::<I>();

            common::init_tensor_device_data::<I>(&self.stream, ctx.get_output(0)?)?;

            let output = ctx.get_output(0)?;

            debug!(
                "[output][dtype={:?}][shape={:?}][stride={:?}]",
                output.dtype(),
                output.shape(),
                output.stride()
            );

            let mut output_dev_ptr = output.try_dev_data_ptr_mut()?;
            let mut output_dev_data = output_dev_ptr.data_mut::<I>();

            unsafe {
                rmlk_cuda::kernels::reduce_mean::compute(
                    self.stream.clone(),
                    func,
                    reduced_dim_prod,
                    axes,
                    rank,
                    info,
                    &input_dev_data,
                    &mut output_dev_data,
                )?;
            }
        } else {
            self.copy_input_to_output::<I>(ctx)?;
        }

        common::write_results_reduce_mean::<I, i64>(
            "debugging/reduce_mean",
            self.stream.clone(),
            ctx,
        )?;

        /*self.stream
            .synchronize()
            .map_err(|e| InternalError::Device { error: e.into() })?;*/

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
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
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
