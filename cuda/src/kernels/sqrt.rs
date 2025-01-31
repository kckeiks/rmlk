use crate::error::Result;
use crate::ptx::SQRT;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};

pub const MODULE_NAME: &str = "sqrt";
pub const FWD_FN_NAMES: [&'static str; 3] = ["sqrt_fwd_f16", "sqrt_fwd_f32", "sqrt_fwd_f64"];
pub const PTX_SRC: &str = SQRT;

/// Launches a CUDA kernel that performs an element-wise square root.
///
/// Panics if the input and output slice are not equal in size.
pub unsafe fn compute<T>(
    func: CudaFunction,
    input_data: &CudaSlice<T>,
    output_data: &mut CudaSlice<T>,
) -> Result<()>
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

#[cfg(test)]
mod test {
    use crate::kernels::sqrt::compute;
    use crate::utils;
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_f32() {
        let device = CudaDevice::new(0).unwrap();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = device.htod_copy(vec![4.0, 9.0, 16.0, 25.0]).unwrap();

        let f = utils::load_kernel(&device.clone(), Op::Sqrt, DataType::Float).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = device
            .alloc_zeros(output_shape.iter().map(|d| *d).product())
            .unwrap();

        unsafe {
            compute::<f32>(f, &x_data, &mut out_data).unwrap();
            let result = device.dtoh_sync_copy(&out_data).unwrap();

            assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0])
        }
    }
}
