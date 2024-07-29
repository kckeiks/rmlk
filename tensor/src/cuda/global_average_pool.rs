use crate::error::Error;
use crate::error::Result;
use cudarc::cudnn;
use cudarc::cudnn::{CudnnDataType, PoolingForward};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use std::ops::AddAssign;
use std::sync::Arc;

pub fn compute_output_shape<T: Num + Copy + AddAssign>(
    x_shape: &[T],
    y_shape: &mut [T],
) -> Result<()> {
    if x_shape.len() < 4 {
        return Err(Error::InvalidInputShapes);
    }

    if x_shape.len() != y_shape.len() {
        return Err(Error::InvalidBufferSize);
    }

    for i in 0..2 {
        y_shape[i] = x_shape[1];
    }

    // For reference, see https://github.com/onnx/onnx/blob/main/docs/Operators.md#outputs-59.
    for i in 2..y_shape.len() {
        y_shape[i] = T::one();
    }

    Ok(())
}

pub fn compute<T>(
    device: Arc<CudaDevice>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
    y_shape: &[i32],
    y_stride: &[i32],
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = cudnn::Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;

    debug_assert!(x_shape.len() == 4 || x_shape.len() == 5);

    let kernel_shape = x_shape[2..].as_ref();
    let pads = x_shape[2..].iter().map(|_| 0).collect::<Box<[i32]>>();
    let strides = x_shape[2..].iter().map(|_| 1).collect::<Box<[i32]>>();

    debug!(
        "x_shape={x_shape:?},\
        kernel_shape={kernel_shape:?},\
        pads={pads:?},\
        strides={strides:?},\
        out_shape={y_shape:?},\
        out_stride={y_stride:?}"
    );

    let x_desc = cudnn
        .create_nd_tensor::<T>(x_shape, x_stride)
        .map_err(|_| Error::CudnnInternal)?;

    let pooling = cudnn
        .create_poolingnd::<T>(
            &kernel_shape,
            &pads,
            &strides,
            cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
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
        .unwrap();

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::cuda::global_average_pool::{compute, compute_output_shape};
    use crate::Tensor;
    use cudarc::driver::CudaDevice;
    use rmlk_ir::DataType;

    #[test]
    fn test_global_average_pool_f32_2d() {
        let device = CudaDevice::new(0).unwrap();

        let x = Tensor::<Vec<()>>::new_with_shape(DataType::Float, vec![1, 1, 3, 3]);
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_data = device
            .htod_copy(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();

        let mut y_shape = x.shape().clone();
        compute_output_shape(x.shape(), y_shape.as_mut_slice()).unwrap();

        let y = Tensor::<Vec<()>>::new_with_shape(DataType::Float, y_shape.to_vec());
        let y_stride = y.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let y_shape = y_shape.iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let mut y_data = device
            .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
            .unwrap();

        compute::<f32>(
            device.clone(),
            (1.0, 0.0),
            &x_data,
            &x_shape,
            &x_stride,
            &mut y_data,
            &y_shape,
            &y_stride,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&y_data).unwrap();

        assert_eq!(result, vec![5.0])
    }
}
