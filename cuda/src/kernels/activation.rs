use crate::error::Result;
use cudarc::cudnn::{sys, ActivationForward, Cudnn, CudnnDataType};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub fn compute<T: CudnnDataType>(
    device: Arc<CudaDevice>,
    (alpha, beta): (T, T),
    x_data: &CudaSlice<T>,
    x_shape: &[i32],
    x_stride: &[i32],
    y_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let cudnn = Cudnn::new(device.clone())?;

    let x_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let y_desc = cudnn.create_nd_tensor::<T>(x_shape, x_stride)?;

    let activation_desc = cudnn.create_activation::<T>(
        sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
        sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        f64::MAX,
    )?;

    let op = ActivationForward {
        act: &activation_desc,
        x: &x_desc,
        y: &y_desc,
    };

    unsafe {
        op.launch((alpha, beta), x_data, y_data)?;
    }

    Ok(())
}

// #[cfg(test)]
// mod test {
//     use crate::kernels::activation::compute;
//     use crate::utils;
//     use cudarc::driver::CudaDevice;
//
//     #[test]
//     fn test_relu_f32() {
//         let device = CudaDevice::new(0).unwrap();
//
//         let x_shape = vec![1, 1, 2, 2];
//         let mut x_stride = vec![0; x_shape.len()];
//         utils::calculate_stride(&x_shape, &mut x_stride);
//
//         let x_data = device.htod_copy(vec![-1.0, 2.0, -3.0, 100.0]).unwrap();
//         let mut y_data = device
//             .alloc_zeros(x_shape.iter().map(|d| *d as usize).product())
//             .unwrap();
//
//         compute::<f32>(
//             device.clone(),
//             (1.0, 0.0),
//             &x_data,
//             &x_shape,
//             &x_stride,
//             &mut y_data,
//         )
//         .unwrap();
//
//         let result = device.dtoh_sync_copy(&y_data).unwrap();
//
//         assert_eq!(result, vec![0.0, 2.0, 0.0, 100.0])
//     }
// }
