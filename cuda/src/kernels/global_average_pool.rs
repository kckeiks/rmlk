use crate::error::Error;
use crate::error::Result;
use cudarc::cudnn;
use cudarc::cudnn::{CudnnDataType, PoolingForward};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::trace;
use num_traits::Num;
use std::fmt::Debug;
use std::ops::AddAssign;
use std::sync::Arc;

pub fn compute_output_shape<T: AddAssign + Copy + Debug + Num>(
    x_shape: &[T],
    y_shape: &mut [T],
) -> Result<()> {
    if x_shape.len() < 4 {
        return Err(Error::InvalidArguments(format!(
            "invalid shape`{:?}` for x",
            x_shape
        )));
    }

    if x_shape.len() != y_shape.len() {
        return Err(Error::InvalidArguments(format!(
            "the shape of x ({:?}) and y ({:?}) do not match",
            x_shape, y_shape
        )));
    }

    for i in 0..2 {
        y_shape[i] = x_shape[i];
    }

    // For reference, see https://github.com/onnx/onnx/blob/main/docs/Operators.md#outputs-59.
    for i in 2..y_shape.len() {
        y_shape[i] = T::one();
    }

    Ok(())
}

pub fn compute<T>(
    stream: Arc<CudaStream>,
    (alpha, beta): (T, T),
    pads: &[i32],
    strides: &[i32],
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    kernel_shape: &[i32],
    y_data: &mut CudaSlice<T>,
    y_shape: &[i32],
    y_stride: &[i32],
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = cudnn::Cudnn::new(stream.clone())?;

    debug_assert!(x_shape.len() == 4 || x_shape.len() == 5);
    debug_assert!(kernel_shape.len() == 2 || kernel_shape.len() == 3);

    trace!(
        "x_shape={x_shape:?},\
        kernel_shape={kernel_shape:?},\
        pads={pads:?},\
        strides={strides:?},\
        out_shape={y_shape:?},\
        out_stride={y_stride:?}"
    );

    let x_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let pooling = cudnn.create_poolingnd::<T>(
        &kernel_shape,
        &pads,
        &strides,
        cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
    )?;

    let out_desc = cudnn.create_nd_tensor(y_shape, y_stride)?;

    let forward_f = PoolingForward {
        pooling: &pooling,
        x: &x_desc,
        y: &out_desc,
    };

    unsafe {
        forward_f.launch((alpha, beta), x_data, y_data)?;
    }

    Ok(())
}
