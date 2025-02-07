use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils::DataIterator;
use crate::{attributes, utils};
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::fmt::Debug;
use std::sync::Arc;

pub struct GatherBackend {
    device: Arc<CudaDevice>,
}

impl GatherBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

impl GatherBackend {
    fn compute_output_shape(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()> {
        let data = ctx.get_input(0)?;
        let indices = ctx.get_input(1)?;

        let data_rank = data.shape().len();
        let indices_rank = indices.shape().len();

        if data_rank < axis {
            return Err(InternalError::InvalidAxis { axis: axis as i64 });
        }

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let output_shape_buf = alloc.allocate::<usize>(data_rank - 1 + indices_rank)?;

        output_shape_buf[..axis].copy_from_slice(&data.shape()[..axis]);
        output_shape_buf[axis..axis + indices_rank].copy_from_slice(&indices.shape());

        if axis + 1 < data_rank {
            output_shape_buf[axis + indices_rank..].copy_from_slice(&data.shape()[axis + 1..]);
        }

        let output = ctx.get_output(0)?;
        let dst_id = output.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape_buf, dst_id)?;

        Ok(())
    }

    fn perform_device_gather<D, I, K>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num + Copy + Debug,
        i64: From<I>,
        K: GatherDeviceProcessor,
    {
        let data = ctx.get_input(0)?;
        let indices = ctx.get_input(1)?;
        let output = ctx.get_output(0)?;

        let data_ptr = data.try_dev_data_ptr()?;
        let indices_ptr = indices.try_dev_data_ptr()?;
        let mut output_ptr = output.try_dev_data_ptr_mut()?;

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let view = indices_ptr.data::<I>();
        let indices_on_host = alloc.allocate_fill::<I>(view.len(), I::zero())?;
        self.device
            .dtoh_sync_copy_into(view.as_ref(), indices_on_host)
            .map_err(rmlk_cuda::Error::from)?;

        let dev_data = data_ptr.data::<D>();
        let mut dev_output = output_ptr.data_mut::<D>();
        K::compute::<I, D>(
            self.device.clone(),
            axis,
            data.shape(),
            data.stride(),
            indices_on_host,
            &dev_data,
            &mut dev_output,
        )?;

        Ok(())
    }

    fn compute_gather<D, K>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: GatherDeviceProcessor,
    {
        // The `indices` input could be `i32` or `i64`.
        let dtype = ctx.get_input(1)?.try_dev_data_ptr()?.dtype();

        match dtype {
            DataType::Int32 => {
                self.perform_device_gather::<D, i32, K>(axis, ctx)?;
            }
            DataType::Int64 => {
                self.perform_device_gather::<D, i64, K>(axis, ctx)?;
            }
            _ => {
                return Err(InternalError::UnsupportedOpForDataType {
                    dtype,
                    op: Op::Gather,
                })
            }
        }

        Ok(())
    }

    fn run_gather<D, K>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: GatherDeviceProcessor,
    {
        let axis = ctx
            .get_attributes()
            .map(attributes::gather::get_axis)
            .unwrap_or(0);

        let data_rank = ctx.get_input(0)?.shape().len();
        let norm_axis = utils::normalize_index(axis as i64, data_rank)?;

        self.compute_output_shape(norm_axis, ctx)?;

        let output = ctx.get_output(0)?;
        common::init_tensor_device_data::<D>(&self.device, output)?;

        self.compute_gather::<D, K>(norm_axis, ctx)?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: GatherDeviceProcessor,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.run_gather::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Gather,
                dtype,
            }),
        }
    }
}

pub trait GatherDeviceProcessor {
    fn compute<I, D>(
        device: Arc<CudaDevice>,
        axis: usize,
        shape: &[usize],
        stride: &[usize],
        indices: &[I],
        data_dev_data: &CudaSlice<D>,
        output_dev_data: &mut CudaSlice<D>,
    ) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr,
        I: Num + Copy,
        i64: From<I>;
}

pub struct DefaultGatherProcessor(());

impl GatherDeviceProcessor for DefaultGatherProcessor {
    fn compute<I, D>(
        device: Arc<CudaDevice>,
        axis: usize,
        shape: &[usize],
        stride: &[usize],
        indices: &[I],
        data_dev_data: &CudaSlice<D>,
        output_dev_data: &mut CudaSlice<D>,
    ) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr,
        I: Num + Copy,
        i64: From<I>,
    {
        // Given `axis` and the `data` shape S=[d_i, d_i2, ..., d_n-1, dn],
        // we split the shape at `axis` to create:
        //
        //              S_lhs = [d_1, d_2, ..., d_axis-1]
        //              S_rhs = [d_axis+1, d_axis+2, ..., d_n]
        //
        // We take sub-slices of size `d_axis+1 * d_axis+2 * ... * d_n` from `data`
        // using `indices`, `axis` and the `data` stride and
        // copy these sub-slices to the output tensor.
        // We must traverse the dimensions `d_axis+1, d_axis+2, ..., d_n`,
        // so we loop `d_1 * d_2 * ... * d_axis-1` times while advancing by
        // section of size `d_axis+1 * d_axis+2 * ... * d_n`.
        let (lhs, _) = shape.split_at(axis);
        // Element count on the dimensions derived from `axis`.
        let elem_count = shape[axis];
        // Count of the slices we will write in the output tensor.
        let mut slice_count = 0;
        // Size of one slice.
        let slice_size = match axis == shape.len() - 1 {
            true => 1,
            false => shape[axis + 1..].iter().product(),
        };
        // We stack these dimensions on the `lhs`.
        let stack_size = lhs.iter().product::<usize>();
        for stack_level in 0..stack_size {
            for dim_i in DataIterator::new(shape, stride, indices) {
                // Get the index.
                let norm_i = utils::normalize_index(i64::from(*dim_i), shape[axis])?;
                let start = stack_level * elem_count + (norm_i * stride[axis]);

                // Slice the input.
                let subslice = data_dev_data.slice(start..start + slice_size);

                // Create a writeable slice of the output.
                let mut out_slice = output_dev_data
                    .slice_mut(slice_count * slice_size..slice_count * slice_size + slice_size);

                // Write to the output slice.
                device.dtod_copy(&subslice, &mut out_slice)?;

                slice_count += 1;
            }
        }

        Ok(())
    }
}

pub struct NoOpGatherProcessor(());

impl GatherDeviceProcessor for NoOpGatherProcessor {
    fn compute<I, D>(
        _: Arc<CudaDevice>,
        _: usize,
        _: &[usize],
        _: &[usize],
        _: &[I],
        _: &CudaSlice<D>,
        _: &mut CudaSlice<D>,
    ) -> Result<()>
    where
        D: CudnnDataType + ValidAsZeroBits + DeviceRepr,
        I: Num + Copy,
        i64: From<I>,
    {
        Ok(())
    }
}
