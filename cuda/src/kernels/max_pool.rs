use crate::error::Error;
use crate::error::Result;
use cudarc::cudnn::{Cudnn, CudnnDataType, PoolingForward};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::{FromPrimitive, Num};
use std::fmt::Debug;
use std::ops::AddAssign;
use std::sync::Arc;

// Todo: figure out how to make this generic.
pub fn compute_output_shape<T>(
    x_shape: &[T],
    kernel_shape: &[T],
    pads: &[T],
    strides: &[T],
    y_shape: &mut [T],
    ceil_mode: bool,
) -> Result<()>
where
    T: AddAssign + Copy + Debug + FromPrimitive + Num,
    f64: From<T>,
{
    let two = T::one() + T::one();
    if kernel_shape.len() == 2 && y_shape.len() == 4 {
        let height =
            (f64::from(x_shape[2] + two * pads[0] - kernel_shape[0]) / f64::from(strides[0])) + 1.0;
        let height = match ceil_mode {
            true => height.ceil(),
            false => height.floor(),
        };

        let width =
            (f64::from(x_shape[3] + two * pads[1] - kernel_shape[1]) / f64::from(strides[1])) + 1.0;
        let width = match ceil_mode {
            true => width.ceil(),
            false => width.floor(),
        };

        y_shape[0] = x_shape[0];
        y_shape[1] = x_shape[1];
        y_shape[2] = T::from_f64(height).ok_or_else(|| {
            Error::InvalidArguments("failed to create a value of type `T` for float64".to_string())
        })?;
        y_shape[3] = T::from_f64(width).ok_or_else(|| {
            Error::InvalidArguments("failed to create a value of type `T` for float64".to_string())
        })?;
    } else if kernel_shape.len() == 3 && y_shape.len() == 5 {
        let depth =
            (f64::from(x_shape[1] + two * pads[0] - kernel_shape[0]) / f64::from(strides[0])) + 1.0;
        let depth = match ceil_mode {
            true => depth.ceil(),
            false => depth.floor(),
        };

        let height =
            (f64::from(x_shape[2] + two * pads[1] - kernel_shape[1]) / f64::from(strides[1])) + 1.0;
        let height = match ceil_mode {
            true => height.ceil(),
            false => height.floor(),
        };

        let width =
            (f64::from(x_shape[3] + two * pads[2] - kernel_shape[2]) / f64::from(strides[2])) + 1.0;
        let width = match ceil_mode {
            true => width.ceil(),
            false => width.floor(),
        };

        y_shape[0] = x_shape[0];
        y_shape[1] = x_shape[1];
        y_shape[2] = T::from_f64(depth).ok_or_else(|| {
            Error::InvalidArguments("failed to create a value of type `T` for float64".to_string())
        })?;
        y_shape[3] = T::from_f64(height).ok_or_else(|| {
            Error::InvalidArguments("failed to create a value of type `T` for float64".to_string())
        })?;
        y_shape[4] = T::from_f64(width).ok_or_else(|| {
            Error::InvalidArguments("failed to create a value of type `T` for float64".to_string())
        })?;
    } else {
        return Err(Error::InvalidArguments(format!(
            "invalid shapes y_shape={y_shape:?} and kernel_shape={kernel_shape:?}"
        )));
    }

    Ok(())
}

pub fn compute<T>(
    stream: Arc<CudaStream>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    kernel_shape: &[i32],
    pads: &[i32],
    strides: &[i32],
    y_data: &mut CudaSlice<T>,
    y_shape: &[i32],
    y_stride: &[i32],
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = Cudnn::new(stream.clone())?;

    let x_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let pooling = cudnn.create_poolingnd::<T>(
        kernel_shape,
        // Todo: Let's preprocess pads.
        pads,
        strides,
        cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
        cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
    )?;

    let out_desc = cudnn.create_nd_tensor(y_shape, y_stride)?;

    let forward_f = PoolingForward {
        pooling: &pooling,
        x: &x_desc,
        y: &out_desc,
    };

    unsafe {
        forward_f
            .launch((alpha, beta), x_data, y_data)
            .map_err(Into::into)
    }
}
