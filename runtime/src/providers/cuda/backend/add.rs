use crate::core::allocators::ScratchAllocator;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::cmp;
use std::sync::Arc;

pub struct AdditionBackend {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl AdditionBackend {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
    }
}

impl AdditionBackend {
    fn process_shapes(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        match a.shape() == b.shape() {
            true => {
                let c = ctx.get_output(0)?;
                let a_index = a.src_id();
                let c_index = c.dst_id();
                ctx.execution_state_mut()
                    .copy_shape_from_within(a_index, c_index)?;
            }
            false => {
                let ndims = cmp::max(a.shape().len(), b.shape().len());
                let alloc = ctx.execution_state().scratch_alloc().clone();
                let c_shape = alloc.allocate_fill(ndims, 0)?;

                if !utils::compute_broadcast_output_shape(a.shape(), b.shape(), c_shape) {
                    let a_id = a.src_id();
                    let b_id = b.src_id();
                    return Err(InternalError::IncompatibleTensorShape {
                        shapes: [
                            (a_id.into(), a.shape().to_vec()),
                            (b_id.into(), b.shape().to_vec()),
                        ]
                        .try_into()
                        .expect("Small map so should succeed"),
                        op: Op::Add,
                    });
                }

                let c = ctx.get_output(0)?;
                let c_index = c.dst_id();
                ctx.execution_state_mut()
                    .copy_shape_from_slice(c_shape, c_index)?;
            }
        }

        Ok(())
    }

    fn compute_addition<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: AdditionKernel,
    {
        // The output should have the same dimensions.
        // We do it now to avoid lifetime errors.
        self.process_shapes(ctx)?;

        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        {
            let c = ctx.get_output(0)?;
            debug!("[a][add][shape={:?}][stride=[{:?}]", a.shape(), a.stride());
            debug!("[b][add][shape={:?}][stride=[{:?}]", b.shape(), b.stride());
            debug!("[c][add][shape={:?}][stride=[{:?}]", c.shape(), c.stride());
        }

        let a_dev_data_ref = a.try_dev_data_ptr()?;
        let a_dev_data = a_dev_data_ref.data::<D>();

        let b_dev_data_ref = b.try_dev_data_ptr()?;
        let b_dev_data = b_dev_data_ref.data::<D>();

        let elem_count: usize = a.shape().iter().product();

        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let mut c = ctx.get_output(0)?;
            let c_dev_data_ref = c.dev_data_ptr_mut();
            let need_to_alloc_dev_data = c_dev_data_ref.is_none()
                || c_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != elem_count)
                    .unwrap_or(true);

            // We need to remove this immutable reference so we can mutate `y`.
            drop(c_dev_data_ref);

            if need_to_alloc_dev_data {
                let c_dev_data = self
                    .device
                    .alloc_zeros::<f32>(c.shape().iter().copied().product::<usize>())
                    .map_err(rmlk_cuda::Error::from)?;
                c.set_dev_data(CudaData::new(c_dev_data));
            };
        }

        let stride_buf_len = cmp::max(a.shape().len(), b.shape().len());
        let strides = ctx
            .execution_state()
            .scratch_alloc()
            .allocate_fill::<usize>(2 * stride_buf_len, 0)?;
        utils::compute_broadcast_stride(a.shape(), b.shape(), a.stride(), b.stride(), strides);
        let (a_stride, b_stride) = strides.split_at(stride_buf_len);

        debug!("[a][add][broadcast][stride={:?}]", a_stride);
        debug!("[b][add][broadcast][stride={:?}]", b_stride);

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let c = ctx.get_output(0)?;
        let mut c_dev_data_ref = c.dev_data_ptr_mut();
        let mut c_dev_data = c_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        let scratch_alloc = &ctx.execution_state().scratch_alloc();

        T::execute::<D>(
            self.device,
            self.f,
            scratch_alloc,
            &a_dev_data,
            &a.shape(),
            a_stride,
            &b_dev_data,
            &b.shape(),
            b_stride,
            c.shape(),
            &mut c_dev_data,
        )?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: AdditionKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_addition::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}

pub trait AdditionKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        alloc: &ScratchAllocator,
        a_dev_data: &CudaSlice<T>,
        a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        b_shape: &[usize],
        b_stride: &[usize],
        c_shape: &[usize],
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl AdditionKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        alloc: &ScratchAllocator,
        a_dev_data: &CudaSlice<T>,
        _a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        _b_shape: &[usize],
        b_stride: &[usize],
        c_shape: &[usize],
        c_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        let ndims = c_shape.len();

        let info_buffer = alloc.allocate(3 * ndims)?;
        info_buffer[..ndims].copy_from_slice(c_shape);
        info_buffer[ndims..2 * ndims].copy_from_slice(a_stride);
        info_buffer[2 * ndims..].copy_from_slice(b_stride);

        unsafe {
            rmlk_cuda::kernels::add::compute(
                device,
                func,
                ndims,
                info_buffer,
                a_dev_data,
                b_dev_data,
                c_dev_data,
            )
            .map_err(Into::into)
        }
    }
}

pub struct NoOpKernel(());

impl AdditionKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: CudaFunction,
        _: &ScratchAllocator,
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &[usize],
        _: &mut CudaSlice<T>,
    ) -> Result<()> {
        Ok(())
    }
}
