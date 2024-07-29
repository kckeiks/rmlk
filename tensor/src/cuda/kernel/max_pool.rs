use crate::error::Error;
use crate::error::Result;
use cudarc::cudnn::{Cudnn, CudnnDataType, PoolingForward};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use num_traits::{FromPrimitive, Num};
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
    T: Num + Copy + AddAssign + FromPrimitive,
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
        y_shape[2] = T::from_f64(height).ok_or(Error::ComputationError)?;
        y_shape[3] = T::from_f64(width).ok_or(Error::ComputationError)?;
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
        y_shape[2] = T::from_f64(depth).ok_or(Error::ComputationError)?;
        y_shape[3] = T::from_f64(height).ok_or(Error::ComputationError)?;
        y_shape[4] = T::from_f64(width).ok_or(Error::ComputationError)?;
    } else {
        return Err(Error::InvalidInputShapes);
    }

    Ok(())
}

pub fn compute<T>(
    device: Arc<CudaDevice>,
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
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;

    let x_desc = cudnn
        .create_nd_tensor::<T>(&x_shape, &x_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let pooling = cudnn
        .create_poolingnd::<T>(
            kernel_shape,
            // Todo: Let's preprocess pads.
            pads,
            strides,
            cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )
        .map_err(|_| Error::CudnnInternal)
        .unwrap();

    let out_desc = cudnn
        .create_nd_tensor(y_shape, y_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let forward_f = PoolingForward {
        pooling: &pooling,
        x: &x_desc,
        y: &out_desc,
    };

    forward_f
        .launch((alpha, beta), x_data, y_data)
        .map_err(|_| Error::CudnnInternal)
}

#[cfg(test)]
mod test {
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::max_pool::{compute, compute_output_shape};
    use crate::{utils, Tensor};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::DataType;

    #[test]
    fn test_max_pool_f32_2d() {
        let device = CudaDevice::new(0).unwrap();

        let x = Tensor::<CudaData>::new_with_shape(DataType::Float, vec![1, 1, 4, 4]);
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_data = device
            .htod_copy(vec![
                1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
            ])
            .unwrap();

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        compute_output_shape(&x_shape, &[2, 2], &[0, 0], &[2, 2], &mut y_shape, false).unwrap();

        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        let mut y_data = device
            .alloc_zeros(y_shape.iter().map(|d| *d as usize).product())
            .unwrap();

        compute::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            &[2, 2],
            &[0, 0],
            &[2, 2],
            &mut y_data,
            &y_shape,
            &y_stride,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 3.0, 4.0])
    }
}
