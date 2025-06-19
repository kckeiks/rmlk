use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils::DataIterator;
use crate::{attributes, utils};
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::{debug, trace};
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

pub struct GatherBackend {
    stream: Arc<CudaStream>,
}

impl GatherBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl GatherBackend {
    fn compute_output_shape(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()> {
        let data = ctx.get_input(0)?;

        debug!(
            "[data][shape={:?}][stride={:?}]",
            data.shape(),
            data.stride()
        );

        let indices = ctx.get_input(1)?;

        debug!(
            "[indices][shape={:?}][stride={:?}]",
            indices.shape(),
            indices.stride()
        );

        let data_rank = data.shape().len();
        let indices_rank = indices.shape().len();

        if data_rank < axis {
            return Err(GatherError::RankAxisMismatch.into());
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

    fn perform_device_gather<D, I>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + Copy + Debug,
        i64: From<I>,
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
        self.stream
            .memcpy_dtoh(view.as_ref(), indices_on_host)
            .map_err(rmlk_cuda::Error::from)?;

        let dev_data = data_ptr.data::<D>();
        let mut dev_output = output_ptr.data_mut::<D>();
        compute::<I, D>(
            self.stream.clone(),
            axis,
            data.shape(),
            data.stride(),
            indices_on_host,
            &dev_data,
            &mut dev_output,
        )?;

        Ok(())
    }

    fn compute_gather<D>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        // The `indices` input could be `i32` or `i64`.
        let dtype = ctx.get_input(1)?.try_dev_data_ptr()?.dtype();

        match dtype {
            DataType::Int32 => {
                self.perform_device_gather::<D, i32>(axis, ctx)?;
            }
            DataType::Int64 => {
                self.perform_device_gather::<D, i64>(axis, ctx)?;
            }
            _ => {
                return Err(InternalError::UnsupportedDataType { dtype }.into());
            }
        }

        Ok(())
    }

    fn run_gather<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let axis = ctx
            .get_attributes()
            .map(|attrs| attributes::gather::get_axis(&attrs))
            .unwrap_or(0);

        debug!("[axis={axis}]");

        let data_rank = ctx.get_input(0)?.shape().len();

        // Todo: handle this conversion better.
        let norm_axis = utils::normalize_index(axis as i64, data_rank)?;

        self.compute_output_shape(norm_axis, ctx)?;

        let output = ctx.get_output(0)?;

        debug!(
            "[output][shape={:?}][stride={:?}]",
            output.shape(),
            output.stride()
        );

        common::init_tensor_device_data::<D>(&self.stream, output)?;

        self.compute_gather::<D>(norm_axis, ctx)?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.run_gather::<f16>(ctx),
            DataType::Float => self.run_gather::<f32>(ctx),
            DataType::Double => self.run_gather::<f64>(ctx),
            DataType::Int32 => self.run_gather::<i32>(ctx),
            DataType::Uint32 => self.run_gather::<u32>(ctx),
            DataType::Int64 => self.run_gather::<i64>(ctx),
            DataType::Uint64 => self.run_gather::<u64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}
fn compute<Indices, Data>(
    stream: Arc<CudaStream>,
    axis: usize,
    shape: &[usize],
    stride: &[usize],
    indices: &[Indices],
    data_dev_data: &CudaSlice<Data>,
    output_dev_data: &mut CudaSlice<Data>,
) -> Result<()>
where
    Data: ValidAsZeroBits + DeviceRepr,
    Indices: Num + Copy + Debug,
    i64: From<Indices>,
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
    let batch_count = shape[..axis].iter().product::<usize>();
    // Fixed offset per batch used to index into the data tensor.
    let batch_offset = stride[..axis].iter().product::<usize>();
    // Count of the slices we will write in the output tensor.
    let mut slice_count = 0;
    // Size of one slice.
    let batch_size = match axis == shape.len() - 1 {
        true => 1,
        false => shape[axis + 1..].iter().product(),
    };
    for batch_index in 0..batch_count {
        for dim_i in DataIterator::new(shape, stride, indices) {
            // Get the index.
            let norm_i = utils::normalize_index(i64::from(*dim_i), shape[axis])?;
            let start = batch_index * batch_offset + (norm_i * stride[axis]);

            trace!("stack_size={batch_count}, stack_level={batch_index}, elem_count={batch_offset}, dim_i={dim_i:?}, norm_i={norm_i}, start={start}, slice_count={slice_count}, slice_size={batch_size}");
            // Slice the input.
            let subslice = data_dev_data.slice(start..start + batch_size);

            // Create a writeable slice of the output.
            let mut out_slice = output_dev_data
                .slice_mut(slice_count * batch_size..slice_count * batch_size + batch_size);

            // Write to the output slice.
            stream.memcpy_dtod(&subslice, &mut out_slice)?;

            slice_count += 1;
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum GatherError {
    RankAxisMismatch,
}

impl Display for GatherError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for GatherError {}
