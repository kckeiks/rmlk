use crate::core::error::InternalError;
use crate::core::Context;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
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
    fn perform_device_gather<T, Tind>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
        Tind: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + Copy + Debug,
        i64: From<Tind>,
    {
        let data_tensor = ctx.get_input(0)?;
        let indices_tensor = ctx.get_input(1)?;

        let indices_len = indices_tensor.payload().len();
        let indices_on_host = ctx
            .execution_state()
            .scratch_alloc()
            .allocate_fill::<Tind>(indices_len, Tind::zero())?;
        indices_tensor.payload_to_host(indices_on_host)?;

        let data_payload = data_tensor.payload();
        let data = data_payload.data::<T>();

        let output_tensor = ctx.get_output(0)?;
        let mut output_payload = output_tensor.payload_mut();
        let mut output_data = output_payload.data_mut::<T>();

        compute::<Tind, T>(
            self.stream.clone(),
            axis,
            &data_tensor.shape(),
            &data_tensor.stride(),
            indices_on_host,
            &data,
            &mut output_data,
        )?;

        Ok(())
    }

    fn compute_gather<T>(&self, axis: usize, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        // The `indices` input could be `i32` or `i64`.
        let dtype = ctx.get_input(1)?.dtype();

        match dtype {
            DataType::Int32 => {
                self.perform_device_gather::<T, i32>(axis, ctx)?;
                #[cfg(feature = "dump")]
                debug::write_results_gather::<T, i32>(
                    "debugging/gather",
                    self.stream.clone(),
                    ctx,
                )?;
            }
            DataType::Int64 => {
                self.perform_device_gather::<T, i64>(axis, ctx)?;
                #[cfg(feature = "dump")]
                debug::write_results_gather::<T, i64>(
                    "debugging/gather",
                    self.stream.clone(),
                    ctx,
                )?;
            }
            _ => {
                return Err(InternalError::UnsupportedDataType { dtype }.into());
            }
        }

        Ok(())
    }

    fn run_gather<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let axis = ctx
            .get_attributes()
            .map(|attrs| attributes::gather::get_axis(&attrs))
            .unwrap_or(0);

        debug!("[axis={axis}]");

        let data_rank = ctx.get_input(0)?.shape().len();

        // Todo: handle this conversion better.
        let norm_axis = utils::normalize_index(axis as i64, data_rank)?;

        compute_output_shape(norm_axis, ctx)?;

        let output = ctx.get_output(0)?;
        let indices_tensor = ctx.get_input(1)?;
        if data_rank == 1 && indices_tensor.is_scalar() {
            output.init_scalar_payload::<T>()?;
        } else {
            output.init_payload::<T>()?;
        }

        debug!(
            "[output][dtype={:?}][shape={:?}][stride={:?}]",
            output.dtype(),
            output.shape(),
            output.stride()
        );

        self.compute_gather::<T>(norm_axis, ctx)?;

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

fn compute_output_shape(axis: usize, ctx: &Context<Cuda>) -> Result<()> {
    let data = ctx.get_input(0)?;

    debug!(
        "[data][dtype={:?}][shape={:?}][stride={:?}]",
        data.dtype(),
        data.shape(),
        data.stride()
    );

    if data.is_scalar() {
        return Err(GatherError::ScalarInputDataNotAllowed.into());
    }

    let indices = ctx.get_input(1)?;

    debug!(
        "[indices][dtype={:?}][shape={:?}][stride={:?}]",
        indices.dtype(),
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
    output.copy_shape_from_slice(output_shape_buf);

    Ok(())
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
            assert!(!output_dev_data.is_empty());
            let mut out_slice = output_dev_data
                .slice_mut(slice_count * batch_size..slice_count * batch_size + batch_size);

            // Write to the output slice.
            stream
                .memcpy_dtod(&subslice, &mut out_slice)
                .map_err(|e| InternalError::Device { error: e.into() })?;

            slice_count += 1;
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum GatherError {
    RankAxisMismatch,
    ScalarInputDataNotAllowed,
}

impl Display for GatherError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GatherError::RankAxisMismatch => write!(f, "rank axis mismatch"),
            GatherError::ScalarInputDataNotAllowed => write!(f, "scalar input data not allowed"),
        }
    }
}

impl std::error::Error for GatherError {}
