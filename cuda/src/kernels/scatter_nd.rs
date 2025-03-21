use crate::ptx::SCATTER_ND;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "scatter_nd";
pub const FWD_FN_NAMES: &[&str] = &[
    "scatter_nd_fwd_f16",
    "scatter_nd_add_fwd_f16",
    "scatter_nd_mul_fwd_f16",
    "scatter_nd_max_fwd_f16",
    "scatter_nd_min_fwd_f16",
    "scatter_nd_fwd_f32",
    "scatter_nd_add_fwd_f32",
    "scatter_nd_mul_fwd_f32",
    "scatter_nd_max_fwd_f32",
    "scatter_nd_min_fwd_f32",
    "scatter_nd_fwd_f64",
    "scatter_nd_add_fwd_f64",
    "scatter_nd_mul_fwd_f64",
    "scatter_nd_max_fwd_f64",
    "scatter_nd_min_fwd_f64",
    "scatter_nd_fwd_i32",
    "scatter_nd_add_fwd_i32",
    "scatter_nd_mul_fwd_i32",
    "scatter_nd_max_fwd_i32",
    "scatter_nd_min_fwd_i32",
    "scatter_nd_fwd_i64",
    "scatter_nd_add_fwd_i64",
    "scatter_nd_mul_fwd_i64",
    "scatter_nd_max_fwd_i64",
    "scatter_nd_min_fwd_i64",
];
pub const PTX_SRC: &str = SCATTER_ND;

pub unsafe fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    num_idx_tuples: usize,
    data_rank: usize,
    indices_rank: usize,
    updates_rank: usize,
    info: &[usize],
    indices: &CudaSlice<T>,
    updates: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    error: &mut CudaSlice<i32>,
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    assert!(data_rank > 0, "data rank must be greater than 0");
    assert!(indices_rank > 0, "indices rank must be greater than 0");
    assert_eq!(
        updates_rank,
        indices_rank + data_rank - info[data_rank + indices_rank - 1] - 1,
        "Rank of updates must be = indices.rank + data.rank - indices.shape[-1] - 1"
    );
    assert!(
        info[2 * data_rank + indices_rank - 1] <= data_rank,
        "indices.shape[-1] must be at most equal to data.rank"
    );
    assert_eq!(
        info[2 * data_rank..2 * data_rank + indices_rank - 1],
        info[2 * data_rank + 2 * indices_rank..2 * data_rank + 2 * indices_rank + indices_rank - 1],
        "indices.shape[..indices.rank - 1] must equal update.shape[..indices.rank - 1]"
    );
    assert_eq!(
        info[2 * data_rank + 2 * indices_rank + indices_rank - 1
            ..data_rank + 2 * indices_rank + updates_rank],
        info[info[data_rank + indices_rank - 1]..data_rank],
        "update.shape[indices.rank - 1..] must be equal to data.shape[indices.shape[-1]..]"
    );
    assert!(
        (indices_rank >= 2 && num_idx_tuples == info[data_rank..indices_rank - 1].iter().product())
            || (indices_rank == 1 && num_idx_tuples == 1),
        "invalid value for `num_idx_tuples`"
    );

    // The CUDA kernel does not need the data's shape.
    let info = device.htod_copy(info[data_rank..].to_vec())?;

    let num_threads = 128;
    let num_blocks = (num_idx_tuples + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        num_idx_tuples,
        output.len(),
        data_rank,
        indices_rank,
        updates_rank,
        &info,
        indices,
        updates,
        output,
        error,
    );

    unsafe { func.launch(config, params)? };

    Ok(())
}
