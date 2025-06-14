use crate::core::error::Result;
use crate::core::{Context, Tensor};
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use rmlk_schema::DataTypeMap;
use std::sync::Arc;

pub fn init_tensor_device_data<T>(
    stream: &Arc<CudaStream>,
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
    T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
{
    let dev_data = stream.alloc_zeros::<T>(0).map_err(rmlk_cuda::Error::from)?;
    tensor.set_dev_data(CudaData::new(dev_data));

    Ok(())
}

pub fn copy_tensor_dev_data<T>(
    stream: &Arc<CudaStream>,
    src: &Tensor<CudaData>,
    dst: &Tensor<CudaData>,
) -> Result<()>
where
    T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let src_dev_ptr = src.try_dev_data_ptr()?;
    let src = src_dev_ptr.data::<T>();

    let mut dst_dev_ptr = dst.try_dev_data_ptr_mut()?;
    let mut dst = dst_dev_ptr.data_mut::<T>();

    stream.memcpy_dtod(src.as_ref(), dst.as_mut())?;

    Ok(())
}

pub fn copy_tensor_shape(
    ctx: &mut Context<Cuda>,
    src: Tensor<CudaData>,
    dst: Tensor<CudaData>,
) -> Result<()> {
    let src_id = src.src_id();
    let dst_id = dst.dst_id();
    ctx.execution_state_mut()
        .copy_shape_from_within(src_id, dst_id)
}
