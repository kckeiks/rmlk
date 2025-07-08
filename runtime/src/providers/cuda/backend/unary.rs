use crate::core::Context;

use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_schema::DataTypeMap;
use std::sync::Arc;

pub unsafe fn compute<T>(
    op: &'static str,
    stream: Arc<CudaStream>,
    f: CudaFunction,
    ctx: &Context<Cuda>,
) -> Result<()>
where
    T: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
{
    let input_tensor = ctx.get_input(0)?;

    debug!(
        "[input][{op}][dtype={:?}][shape={:?}][stride=[{:?}]",
        input_tensor.dtype(),
        input_tensor.shape(),
        input_tensor.stride()
    );

    let output_tensor = ctx.get_output(0)?;
    output_tensor.copy_shape(input_tensor.shape_handle());
    output_tensor.init_payload::<T>()?;

    debug!(
        "[output][{op}][dtype={:?}][shape={:?}][stride=[{:?}]",
        output_tensor.dtype(),
        output_tensor.shape(),
        output_tensor.stride()
    );

    let input_payload = input_tensor.payload();
    let input_data = input_payload.data::<T>();

    let mut output_payload = output_tensor.payload_mut();
    let mut output_data = output_payload.data_mut();

    rmlk_cuda::kernels::unary::compute::<T>(stream.clone(), f, &input_data, &mut output_data)?;

    Ok(())
}
