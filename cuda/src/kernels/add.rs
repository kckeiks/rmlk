use crate::error::Error;
use crate::error::Result;
use crate::ptx::BINARY_ADD;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "binary_add";
pub const FWD_FN_NAMES: [&'static str; 3] = ["badd_fwd_f16", "badd_fwd_f32", "badd_fwd_f64"];
pub const PTX_SRC: &str = BINARY_ADD;

pub fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    lhs_data: &CudaSlice<T>,
    lhs_shape: &[usize],
    lhs_stride: &[usize],
    rhs_data: &CudaSlice<T>,
    rhs_shape: &[usize],
    rhs_stride: &[usize],
    out_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    // Todo: more assertions here.
    debug_assert!(lhs_shape == rhs_shape);

    // Todo: Validate that the tensors are valid for the operation.
    // Todo: should we directly initialize this in the device?
    // Todo: pass in vector.
    let mut info: Vec<usize> = Vec::with_capacity(3 * lhs_shape.len());
    info.extend(lhs_shape);
    info.extend(lhs_stride);
    info.extend(rhs_stride);

    let info = device
        .htod_copy(info.as_slice().to_vec())
        .map_err(|_| Error::Unknown)?;

    let elem_count: usize = lhs_shape.iter().product();
    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        elem_count,
        lhs_shape.len(),
        &info,
        lhs_data,
        rhs_data,
        out_data,
    );

    unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::kernels::add::compute;
    use crate::utils;
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_add_f32() {
        let device = CudaDevice::new(0).unwrap();

        let lhs_shape = vec![4, 1, 1, 1];
        let mut lhs_stride = vec![0; lhs_shape.len()];
        utils::calculate_stride(&lhs_shape, &mut lhs_stride);
        let lhs_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let rhs_shape = vec![4, 1, 1, 1];
        let mut rhs_stride = vec![0; rhs_shape.len()];
        utils::calculate_stride(&rhs_shape, &mut rhs_stride);
        let rhs_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let f = utils::load_kernel(device.clone(), Op::Add, DataType::Float).unwrap();

        let mut out_data = device
            .alloc_zeros(lhs_shape.iter().map(|d| *d as usize).product())
            .unwrap();

        compute::<f32>(
            device.clone(),
            f,
            &lhs_data,
            &lhs_shape,
            &lhs_stride,
            &rhs_data,
            &rhs_shape,
            &rhs_stride,
            &mut out_data,
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }
}
