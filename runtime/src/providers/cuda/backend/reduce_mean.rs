use crate::attributes::reduce_mean;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct ReduceMeanBackend {
    device: Arc<CudaDevice>,
    kernel: CudaFunction,
}

impl ReduceMeanBackend {
    pub fn new(device: Arc<CudaDevice>, kernel: CudaFunction) -> Self {
        Self { device, kernel }
    }

    fn compute_output_shape(&mut self, axes: &[usize], ctx: &mut Context<Cuda>) -> Result<()> {
        let input = ctx.get_input(0)?;
        let rank = input.shape().len();

        let alloc = ctx.execution_state().scratch_alloc().clone();
        let reduced = alloc.allocate::<bool>(rank)?;

        for axis in axes.iter().copied() {
            if axis > rank {
                return Err(InternalError::InvalidInput {
                    input: 1,
                    op: Op::ReduceMean,
                    message: "`axis` value cannot be larger than the rank".to_string(),
                });
            }

            if reduced[axis] {
                return Err(InternalError::InvalidInput {
                    input: 1,
                    op: Op::ReduceMean,
                    message: "`axes` cannot contain duplicate values".to_string(),
                });
            }

            reduced[axis] = true;
        }

        let keep_dims = ctx
            .get_attributes()
            .map(reduce_mean::get_keep_dims)
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
                return Err(InternalError::InvalidInput {
                    input: 1,
                    op: Op::ReduceMean,
                    message: "`axes` length cannot be larger than the rank".to_string(),
                });
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
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
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

    fn compute_reduce_mean<I, K>(mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: ReduceMeanKernel,
    {
        let rank = ctx.get_input(0)?.shape().len();
        let alloc = ctx.execution_state().scratch_alloc().clone();

        let noop_with_empty_axes = ctx
            .get_attributes()
            .map(reduce_mean::get_noop_with_empty_axes)
            .unwrap_or(false);

        // We either return a non-empty axes list or nothing.
        let axes = {
            match ctx.get_input(1).ok() {
                Some(tensor) => {
                    let axes_dev_ptr = tensor.try_dev_data_ptr()?;
                    let axes_dev_data = axes_dev_ptr.data::<usize>();

                    if axes_dev_data.len() > 0 {
                        let buf = alloc.allocate::<usize>(axes_dev_data.len())?;
                        self.device
                            .dtoh_sync_copy_into(axes_dev_data.as_ref(), buf)
                            .map_err(rmlk_cuda::Error::from)?;
                        Some(buf)
                    } else {
                        None
                    }
                }
                None => {
                    let buf = alloc.allocate::<usize>(rank)?;
                    for i in 0..rank {
                        buf[i] = i;
                    }
                    Some(buf)
                }
            }
        };

        if (axes.is_none() && !noop_with_empty_axes) || axes.is_some() {
            let axes = match axes {
                None => {
                    let buf = alloc.allocate::<usize>(rank)?;
                    for i in 0..rank {
                        buf[i] = i;
                    }
                    buf
                }
                Some(axes) => axes,
            };

            self.compute_output_shape(&axes, ctx)?;

            let input = ctx.get_input(0)?;

            let info = alloc.allocate(2 * rank)?;
            info[..rank].copy_from_slice(input.shape());
            info[rank..2 * rank].copy_from_slice(input.stride());

            let input_dev_ptr = input.try_dev_data_ptr()?;
            let input_dev_data = input_dev_ptr.data::<I>();

            let output = ctx.get_output(0)?;
            let mut output_dev_ptr = output.try_dev_data_ptr_mut()?;
            let mut output_dev_data = output_dev_ptr.data_mut::<I>();

            K::execute(
                self.device.clone(),
                self.kernel,
                0,
                axes,
                rank,
                info,
                &input_dev_data,
                &mut output_dev_data,
            )?;
        } else {
            self.copy_input_to_output::<I>(ctx)?;
        }

        Ok(())
    }

    pub fn compute<K>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        K: ReduceMeanKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_reduce_mean::<f32, K>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::ReduceMean,
                dtype,
            }),
        }
    }
}

pub trait ReduceMeanKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        reduced_dim_prod: usize,
        axes: &[usize],
        rank: usize,
        tensor_info: &[usize],
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl ReduceMeanKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        reduced_dim_prod: usize,
        axes: &[usize],
        rank: usize,
        tensor_info: &[usize],
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        unsafe {
            rmlk_cuda::kernels::reduce_mean::compute(
                device,
                func,
                reduced_dim_prod,
                axes,
                rank,
                tensor_info,
                input,
                output,
            )
            .map_err(Into::into)
        }
    }
}
