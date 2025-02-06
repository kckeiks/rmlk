use crate::core::error::Result;
use crate::core::Tensor;
use crate::providers::cuda::data::CudaData;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::DataTypeMap;
use std::sync::Arc;

pub fn init_tensor_device_data<T>(
    device: &Arc<CudaDevice>,
    mut tensor: Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
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
        let dev_data = device
            .alloc_zeros::<f32>(size)
            .map_err(rmlk_cuda::Error::from)?;
        tensor.set_dev_data(CudaData::new(dev_data));
    };

    Ok(())
}
