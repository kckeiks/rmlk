use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use std::sync::Arc;

/// Launches a CUDA kernel that performs an element-wise unary operation.
///
/// Panics if the input and output slice are not equal in size.
pub unsafe fn compute<T>(
    stream: Arc<CudaStream>,
    func: CudaFunction,
    input_data: &CudaSlice<T>,
    output_data: &mut CudaSlice<T>,
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(input_data.len(), output_data.len());

    let elem_count = output_data.len();

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&elem_count)
            .arg(input_data)
            .arg(output_data)
            .launch(config)?;
    };

    Ok(())
}

/// Launches a CUDA kernel that performs an element-wise unary operation.
///
/// Panics if the input and output slice are not equal in size.
pub unsafe fn explicit_io_types_compute<Src, Dst>(
    stream: &Arc<CudaStream>,
    func: CudaFunction,
    input_data: &CudaSlice<Src>,
    output_data: &mut CudaSlice<Dst>,
) -> crate::error::Result<()>
where
    Src: ValidAsZeroBits + DeviceRepr,
    Dst: ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(input_data.len(), output_data.len());

    let elem_count = output_data.len();

    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&elem_count)
            .arg(input_data)
            .arg(output_data)
            .launch(config)?;
    };

    Ok(())
}
