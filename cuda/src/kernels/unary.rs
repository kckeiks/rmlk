use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};

/// Launches a CUDA kernel that performs an element-wise unary operation.
///
/// Panics if the input and output slice are not equal in size.
pub unsafe fn compute<T>(
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

    let params = (elem_count, input_data, output_data);

    unsafe { func.launch(config, params)? };

    Ok(())
}

/// Launches a CUDA kernel that performs an element-wise unary operation.
///
/// Panics if the input and output slice are not equal in size.
pub unsafe fn explicit_io_types_compute<Src, Dst>(
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

    let params = (elem_count, input_data, output_data);

    unsafe { func.launch(config, params)? };

    Ok(())
}
