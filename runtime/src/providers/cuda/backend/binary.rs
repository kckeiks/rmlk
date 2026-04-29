use crate::core::allocators::ScratchAllocator;
use crate::core::Context;
use crate::providers::cuda;
use crate::providers::cuda::allocator::CudaBump;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use rmlk_cuda::kernels::binary;
use rmlk_schema::DataTypeMap;
use std::cmp;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

pub unsafe fn compute<X, Y, O>(
    op: &'static str,
    stream: Arc<CudaStream>,
    scratch_cuda_alloc: Rc<CudaBump>,
    f: CudaFunction,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    X: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    Y: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    O: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    update_output_shape(ctx)?;

    let a = ctx.get_input(0)?;

    debug!(
        "[a][{op}][dtype={:?}][shape={:?}][stride=[{:?}]",
        a.dtype(),
        a.shape(),
        a.stride()
    );

    let b = ctx.get_input(1)?;

    debug!(
        "[b][{op}][dtype={:?}][shape={:?}][stride=[{:?}]",
        b.dtype(),
        b.shape(),
        b.stride()
    );

    let c_tensor = ctx.get_output(0)?;
    if a.is_scalar() && b.is_scalar() {
        c_tensor.init_scalar_payload::<O>()?;
    } else {
        c_tensor.init_payload::<O>()?;
    }

    debug!(
        "[c][{op}][dtype={:?}][shape={:?}][stride=[{:?}]",
        c_tensor.dtype(),
        c_tensor.shape(),
        c_tensor.stride()
    );

    let a_payload = a.payload();
    let a_data = a_payload.data::<X>();

    let b_payload = b.payload();
    let b_data = b_payload.data::<Y>();

    let (a_shape, a_stride) = cuda::utils::get_kernel_safe_shape_and_stride(&a);
    let (b_shape, b_stride) = cuda::utils::get_kernel_safe_shape_and_stride(&b);

    let stride_buf_len = cmp::max(a_shape.len(), b_shape.len());
    let strides = ctx
        .execution_state()
        .scratch_alloc()
        .allocate_fill::<usize>(2 * stride_buf_len, 0)?;
    utils::compute_broadcast_stride(&a_shape, &b_shape, &a_stride, &b_stride, strides);
    let (a_stride, b_stride) = strides.split_at(stride_buf_len);

    debug!(
        "[a][{op}][broadcast][shape={:?}][stride={:?}]",
        a_shape, a_stride
    );
    debug!(
        "[b][{op}][broadcast][shape={:?}][stride={:?}]",
        b_shape, b_stride
    );

    let (c_shape, _) = cuda::utils::get_kernel_safe_shape_and_stride(&c_tensor);

    let mut c_payload = c_tensor.payload_mut();
    let mut c_dev_data = c_payload.data_mut::<O>();

    let rank = c_shape.len();
    let alloc = ctx.execution_state().scratch_alloc().clone();
    let info_cuda_data =
        create_info_data_on_dev(&scratch_cuda_alloc, &alloc, a_stride, b_stride, &c_shape)?;
    let info_payload = info_cuda_data.data::<usize>();

    binary::compute_with_types::<X, Y, O>(
        stream,
        f,
        rank,
        &info_payload,
        &a_data,
        &b_data,
        &mut c_dev_data,
    )?;

    Ok(())
}

fn create_info_data_on_dev(
    cuda_alloc: &CudaBump,
    host_alloc: &ScratchAllocator,
    a_stride: &[usize],
    b_stride: &[usize],
    c_shape: &[usize],
) -> Result<CudaData> {
    assert_eq!(c_shape.len(), a_stride.len());
    assert_eq!(b_stride.len(), a_stride.len());

    let rank = c_shape.len();
    let info_buffer = host_alloc.allocate(3 * rank)?;
    info_buffer[..rank].copy_from_slice(c_shape);
    info_buffer[rank..2 * rank].copy_from_slice(a_stride);
    info_buffer[2 * rank..].copy_from_slice(b_stride);

    Ok(cuda_alloc.alloc_from_slice_with_fallback(info_buffer)?)
}

fn update_output_shape(ctx: &Context<Cuda>) -> Result<()> {
    let a = ctx.get_input(0)?;
    let b = ctx.get_input(1)?;

    let equal_shape = a.shape().as_ref() == b.shape().as_ref();

    match equal_shape {
        true => {
            let c = ctx.get_output(0)?;
            c.copy_shape(a.shape_handle());
        }
        false => {
            let rank = cmp::max(a.shape().len(), b.shape().len());
            let alloc = ctx.execution_state().scratch_alloc().clone();
            let c_shape = alloc.allocate_fill(rank, 0)?;

            if !utils::compute_broadcast_output_shape(&a.shape(), &b.shape(), c_shape) {
                return Err(BinaryOpError::IncompatibleTensorShape {
                    shapes: [(0, a.shape().to_vec()), (1, b.shape().to_vec())]
                        .try_into()
                        .expect("Small map so should succeed"),
                }
                .into());
            }

            let c = ctx.get_output(0)?;
            c.copy_shape_from_slice(c_shape);
        }
    }

    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum BinaryOpError {
    #[error("incompatible tensor shapes: {shapes:?}")]
    IncompatibleTensorShape { shapes: HashMap<usize, Vec<usize>> },
}
