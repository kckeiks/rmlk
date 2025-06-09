use crate::core::error::Result;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_schema::DataTypeMap;
use std::sync::Arc;

pub unsafe fn compute<I, K>(
    op: &'static str,
    stream: Arc<CudaStream>,
    f: CudaFunction,
    ctx: &mut Context<Cuda>,
) -> Result<()>
where
    I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    K: UnaryKernel,
{
    let input = ctx.get_input(0)?;

    {
        let output = ctx.get_output(0)?;
        debug!(
            "[input][{op}][shape={:?}][stride=[{:?}]",
            input.shape(),
            input.stride()
        );
        debug!(
            "[output][{op}][shape={:?}][stride=[{:?}]",
            output.shape(),
            output.stride()
        );
    }

    common::init_tensor_device_data::<I>(&stream, input)?;

    let input = ctx.get_input(0)?;

    let input_dev_data_ref = input.try_dev_data_ptr()?;
    let input_dev_data = input_dev_data_ref.data::<I>();

    // The device data should exist so we will execute the kernel
    // and update the destination device data with the result.
    let output = ctx.get_output(0)?;
    let mut output_dev_data_ref = output.dev_data_ptr_mut();
    let mut output_dev_data = output_dev_data_ref
        .as_mut()
        .expect("we already checked that it initialized")
        .data_mut();

    let rank = output.shape().len();

    let info_buffer = ctx.execution_state().scratch_alloc().allocate(2 * rank)?;
    info_buffer[..rank].copy_from_slice(output.shape());
    info_buffer[rank..2 * rank].copy_from_slice(input.stride());

    K::execute::<I>(stream.clone(), f, &input_dev_data, &mut output_dev_data)?;

    Ok(())
}

pub trait UnaryKernel {
    fn execute<T>(
        stream: Arc<CudaStream>,
        func: CudaFunction,
        input_dev_data: &CudaSlice<T>,
        output_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}
