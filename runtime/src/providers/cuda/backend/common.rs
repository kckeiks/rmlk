use crate::core::error::InternalError;
use crate::core::Tensor;
use crate::providers::cuda::data::CudaData;
use anyhow::Result;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::DataTypeMap;
use std::sync::Arc;
// Todo: handle scalars.
// Users may want a completely empty tensor but this doesnt do that.
pub fn init_tensor_device_data<T>(
    stream: &Arc<CudaStream>,
    mut tensor: Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let size = tensor.shape().iter().copied().product::<usize>();

    // Allocate device data for the tensor if we haven't done it yet
    // or if the existing allocated data has a different size.
    let dev_data_ref = tensor.dev_data_ptr_mut();
    let need_to_alloc_dev_data = dev_data_ref.is_none()
        || dev_data_ref
            .as_ref()
            .map(|data| data.data::<T>().len() != size)
            .unwrap_or(true);

    // We need to remove this immutable reference so we can mutate `y`.
    drop(dev_data_ref);

    if need_to_alloc_dev_data {
        let dev_data = stream
            .alloc_zeros::<T>(size)
            .map_err(rmlk_cuda::Error::from)?;
        tensor.set_dev_data(CudaData::new(dev_data));
    };

    Ok(())
}

pub fn init_tensor_device_data_with_empty_slice<T>(
    stream: &Arc<CudaStream>,
    mut tensor: Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
{
    let dev_data = stream.alloc_zeros::<T>(0).map_err(rmlk_cuda::Error::from)?;
    tensor.set_dev_data(CudaData::new(dev_data));

    Ok(())
}

pub fn copy_tensor_dev_data<T>(
    stream: &Arc<CudaStream>,
    src: &Tensor<CudaData>,
    dst: &mut Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
{
    let src_dev_ptr = src.try_dev_data_ptr()?;
    let src = src_dev_ptr.data::<T>();

    let mut dev_data = unsafe {
        stream
            .alloc::<T>(src.len())
            .map_err(rmlk_cuda::Error::from)?
    };
    stream
        .memcpy_dtod(src.as_ref(), &mut dev_data)
        .map_err(|e| InternalError::Device { error: e.into() })?;
    dst.set_dev_data(CudaData::new(dev_data));

    Ok(())
}
