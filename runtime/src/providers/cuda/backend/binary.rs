use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use rmlk_cuda::kernels::binary;
use rmlk_schema::DataTypeMap;
use std::cmp;
use std::sync::Arc;

pub unsafe fn compute<X, Y, O>(
    op: &'static str,
    stream: Arc<CudaStream>,
    f: CudaFunction,
    ctx: &mut Context<Cuda>,
) -> Result<()>
where
    X: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    Y: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    O: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    // The output should have the same dimensions.
    // We do it now to avoid lifetime errors.
    process_shapes(ctx)?;

    let a = ctx.get_input(0)?;
    let b = ctx.get_input(1)?;

    debug!("[a][{op}][shape={:?}][stride=[{:?}]", a.shape(), a.stride());
    debug!("[b][{op}][shape={:?}][stride=[{:?}]", b.shape(), b.stride());

    let a_dev_data_ref = a.try_dev_data_ptr()?;
    let a_dev_data = a_dev_data_ref.data::<X>();

    let b_dev_data_ref = b.try_dev_data_ptr()?;
    let b_dev_data = b_dev_data_ref.data::<Y>();

    let c_tensor = ctx.get_output(0)?;

    debug!(
        "[c][{op}][shape={:?}][stride=[{:?}]",
        c_tensor.shape(),
        c_tensor.stride()
    );

    common::init_tensor_device_data::<O>(&stream, c_tensor)?;

    let stride_buf_len = cmp::max(a.shape().len(), b.shape().len());
    let strides = ctx
        .execution_state()
        .scratch_alloc()
        .allocate_fill::<usize>(2 * stride_buf_len, 0)?;
    utils::compute_broadcast_stride(a.shape(), b.shape(), a.stride(), b.stride(), strides);
    let (a_stride, b_stride) = strides.split_at(stride_buf_len);

    debug!("[a][{op}][broadcast][stride={:?}]", a_stride);
    debug!("[b][{op}][broadcast][stride={:?}]", b_stride);

    // The device data should exist so we will execute the kernel
    // and update the destination device data with the result.
    let c = ctx.get_output(0)?;
    let mut c_dev_data_ref = c.dev_data_ptr_mut();
    let mut c_dev_data = c_dev_data_ref
        .as_mut()
        .expect("we already checked that it initialized")
        .data_mut::<O>();

    let rank = c.shape().len();

    let info_buffer = ctx.execution_state().scratch_alloc().allocate(3 * rank)?;
    info_buffer[..rank].copy_from_slice(c.shape());
    info_buffer[rank..2 * rank].copy_from_slice(a_stride);
    info_buffer[2 * rank..].copy_from_slice(b_stride);

    binary::compute_with_types::<X, Y, O>(
        stream,
        f,
        rank,
        info_buffer,
        &a_dev_data,
        &b_dev_data,
        &mut c_dev_data,
    )?;

    Ok(())
}

fn process_shapes(ctx: &mut Context<Cuda>) -> Result<()> {
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
            let rank = cmp::max(a.shape().len(), b.shape().len());
            let alloc = ctx.execution_state().scratch_alloc().clone();
            let c_shape = alloc.allocate_fill(rank, 0)?;

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
                }
                .into());
            }

            let c = ctx.get_output(0)?;
            let c_index = c.dst_id();
            ctx.execution_state_mut()
                .copy_shape_from_slice(c_shape, c_index)?;
        }
    }

    Ok(())
}
