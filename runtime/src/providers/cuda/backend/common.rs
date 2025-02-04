use std::sync::Arc;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::DataTypeMap;
use crate::core::{Tensor};
use crate::providers::cuda::data::CudaData;
use crate::core::error::Result;

pub fn alloc_output_data<T>(device: Arc<CudaDevice>, mut output: Tensor<CudaData>) -> Result<()>
where
    T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,

{
    let size = output.shape().iter().copied().product::<usize>();

    // Allocate device data for the tensor if we haven't done it yet
    // or if the existing allocated data has a different size.
    let output_dev_data_ref = output.dev_data_ptr_mut();
    let need_to_alloc_dev_data = output_dev_data_ref.is_none()
        || output_dev_data_ref
        .as_ref()
        .map(|data| data.data::<T>().len() != size)
        .unwrap_or(true);

    // We need to remove this immutable reference so we can mutate `y`.
    drop(output_dev_data_ref);

    if need_to_alloc_dev_data {
        let output_dev_data = device
            .alloc_zeros::<f32>(size)
            .map_err(rmlk_cuda::Error::from)?;
        output.set_dev_data(CudaData::new(output_dev_data));
    };

    Ok(())
}