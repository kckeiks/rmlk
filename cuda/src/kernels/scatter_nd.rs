use crate::ptx::SCATTER_ND;
use crate::Error;
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

#[derive(Debug, Clone, Copy)]
pub enum ScatterNdKernel {
    FwdF16,
    AddFwdF16,
    MulFwdF16,
    MaxFwdF16,
    MinFwdF16,
    FwdF32,
    AddFwdF32,
    MulFwdF32,
    MaxFwdF32,
    MinFwdF32,
    FwdF64,
    AddFwdF64,
    MulFwdF64,
    MaxFwdF64,
    MinFwdF64,
    FwdI32,
    AddFwdI32,
    MulFwdI32,
    MaxFwdI32,
    MinFwdI32,
    FwdI64,
    AddFwdI64,
    MulFwdI64,
    MaxFwdI64,
    MinFwdI64,
}

impl From<ScatterNdKernel> for &'static str {
    fn from(kernel: ScatterNdKernel) -> Self {
        match kernel {
            ScatterNdKernel::FwdF16 => "scatter_nd_fwd_f16",
            ScatterNdKernel::AddFwdF16 => "scatter_nd_add_fwd_f16",
            ScatterNdKernel::MulFwdF16 => "scatter_nd_mul_fwd_f16",
            ScatterNdKernel::MaxFwdF16 => "scatter_nd_max_fwd_f16",
            ScatterNdKernel::MinFwdF16 => "scatter_nd_min_fwd_f16",
            ScatterNdKernel::FwdF32 => "scatter_nd_fwd_f32",
            ScatterNdKernel::AddFwdF32 => "scatter_nd_add_fwd_f32",
            ScatterNdKernel::MulFwdF32 => "scatter_nd_mul_fwd_f32",
            ScatterNdKernel::MaxFwdF32 => "scatter_nd_max_fwd_f32",
            ScatterNdKernel::MinFwdF32 => "scatter_nd_min_fwd_f32",
            ScatterNdKernel::FwdF64 => "scatter_nd_fwd_f64",
            ScatterNdKernel::AddFwdF64 => "scatter_nd_add_fwd_f64",
            ScatterNdKernel::MulFwdF64 => "scatter_nd_mul_fwd_f64",
            ScatterNdKernel::MaxFwdF64 => "scatter_nd_max_fwd_f64",
            ScatterNdKernel::MinFwdF64 => "scatter_nd_min_fwd_f64",
            ScatterNdKernel::FwdI32 => "scatter_nd_fwd_i32",
            ScatterNdKernel::AddFwdI32 => "scatter_nd_add_fwd_i32",
            ScatterNdKernel::MulFwdI32 => "scatter_nd_mul_fwd_i32",
            ScatterNdKernel::MaxFwdI32 => "scatter_nd_max_fwd_i32",
            ScatterNdKernel::MinFwdI32 => "scatter_nd_min_fwd_i32",
            ScatterNdKernel::FwdI64 => "scatter_nd_fwd_i64",
            ScatterNdKernel::AddFwdI64 => "scatter_nd_add_fwd_i64",
            ScatterNdKernel::MulFwdI64 => "scatter_nd_mul_fwd_i64",
            ScatterNdKernel::MaxFwdI64 => "scatter_nd_max_fwd_i64",
            ScatterNdKernel::MinFwdI64 => "scatter_nd_min_fwd_i64",
        }
    }
}

pub unsafe fn compute<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    num_idx_tuples: usize,
    data_rank: usize,
    indices_rank: usize,
    updates_rank: usize,
    info: &[usize],
    indices: &CudaSlice<usize>,
    updates: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    error: &mut CudaSlice<i32>,
) -> crate::error::Result<()>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    let _data_shape = &info[..data_rank];
    let _indices_shape = &info[2 * data_rank..2 * data_rank + indices_rank];
    let _updates_shape =
        &info[2 * data_rank + 2 * indices_rank..2 * data_rank + 2 * indices_rank + updates_rank];

    assert!(data_rank > 0, "data rank must be greater than 0");
    assert!(indices_rank > 0, "indices rank must be greater than 0");
    assert!(
        (indices_rank > 1
            && updates_rank == data_rank + indices_rank - _indices_shape[indices_rank - 1] - 1)
            || (indices_rank == 1
                && updates_rank > 1
                && updates_rank == data_rank + indices_rank - 1)
            || indices_rank == updates_rank,
        "Rank of updates must be = indices.rank + data.rank - indices.shape[-1] - 1"
    );
    assert!(
        (indices_rank > 1 && _indices_shape[indices_rank - 1] <= data_rank)
            || (indices_rank == 1 && indices_rank <= data_rank),
        "indices.shape[-1] must be at most equal to data.rank"
    );
    assert_eq!(
        _indices_shape[..indices_rank - 1],
        _updates_shape[..indices_rank - 1],
        "indices.shape[..indices.rank - 1] must equal update.shape[..indices.rank - 1]"
    );
    assert!(
        (indices_rank > 1
            && _updates_shape[indices_rank - 1..]
                == _data_shape[_indices_shape[indices_rank - 1]..])
            || (indices_rank == 1 && _updates_shape[indices_rank..] == _data_shape[indices_rank..])
            || (indices_rank == updates_rank),
        "update.shape[indices.rank - 1..] must be equal to data.shape[indices.shape[-1]..]"
    );
    assert!(
        (indices_rank > 1 && num_idx_tuples == _indices_shape[..indices_rank - 1].iter().product())
            || (indices_rank == 1 && num_idx_tuples == _indices_shape.iter().product()),
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

pub fn load_kernel(
    device: &Arc<CudaDevice>,
    kernel_name: ScatterNdKernel,
) -> crate::error::Result<CudaFunction> {
    if !device.has_func(MODULE_NAME, kernel_name.into()) {
        device
            .load_ptx(PTX_SRC.into(), MODULE_NAME, FWD_FN_NAMES)
            .map_err(|e| Error::Internal(format!("failed to load kernel: {e:?}")))?
    }

    Ok(device
        .get_func(MODULE_NAME, kernel_name.into())
        .expect("To have been loaded"))
}

#[cfg(test)]
mod test {
    use crate::kernels::scatter_nd::{compute, load_kernel, ScatterNdKernel};
    use crate::utils;
    use cudarc::driver::CudaDevice;

    #[cfg(test)]
    pub fn create_info_buffer(
        data_shape: &[usize],
        data_stride: &[usize],
        indices_shape: &[usize],
        indices_stride: &[usize],
        updates_shape: &[usize],
        updates_stride: &[usize],
    ) -> Vec<usize> {
        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let updates_rank = updates_shape.len();
        let mut info_buffer = vec![0usize; 2 * data_rank + 2 * indices_rank + 2 * updates_rank];

        info_buffer[..data_rank].copy_from_slice(data_shape);
        info_buffer[data_rank..2 * data_rank].copy_from_slice(data_stride);
        info_buffer[2 * data_rank..2 * data_rank + indices_rank].copy_from_slice(indices_shape);
        info_buffer[2 * data_rank + indices_rank..2 * data_rank + 2 * indices_rank]
            .copy_from_slice(indices_stride);
        info_buffer
            [2 * data_rank + 2 * indices_rank..2 * data_rank + 2 * indices_rank + updates_rank]
            .copy_from_slice(updates_shape);
        info_buffer[2 * data_rank + 2 * indices_rank + updates_rank..]
            .copy_from_slice(updates_stride);

        info_buffer
    }

    #[test]
    fn test_update_elements_2d_data() {
        let device = CudaDevice::new(0).unwrap();

        let data_shape = vec![3, 3];
        let mut data_stride = vec![0; data_shape.len()];
        utils::calculate_stride(&data_shape, &mut data_stride);
        let data = vec![0.0; 3 * 3];

        let indices_shape = vec![2, 2];
        let mut indices_stride = vec![0; indices_shape.len()];
        utils::calculate_stride(&indices_shape, &mut indices_stride);
        let indices = device.htod_copy(vec![0, 1, 2, 2]).unwrap();

        let updates_shape = vec![2];
        let mut updates_stride = vec![0; updates_shape.len()];
        utils::calculate_stride(&updates_shape, &mut updates_stride);
        let updates = device.htod_copy(vec![5.0, 8.0]).unwrap();

        let output_shape = data_shape.clone();
        let mut output_stride = vec![0; output_shape.len()];
        utils::calculate_stride(&output_shape, &mut output_stride);
        let mut output = device.htod_copy(data).unwrap();

        let f = load_kernel(&device.clone(), ScatterNdKernel::FwdF32).unwrap();

        let info = create_info_buffer(
            &data_shape,
            &data_stride,
            &indices_shape,
            &indices_stride,
            &updates_shape,
            &updates_stride,
        );

        let mut error = device.alloc_zeros(1).unwrap();

        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let updates_rank = updates_shape.len();

        let num_idx_tuples = indices_shape[0..indices_rank - 1].iter().product();

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                num_idx_tuples,
                data_rank,
                indices_rank,
                updates_rank,
                &info,
                &indices,
                &updates,
                &mut output,
                &mut error,
            )
            .unwrap();
        }

        let result = device.dtoh_sync_copy(&output).unwrap();

        assert_eq!(result, vec![0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0])
    }

    #[test]
    fn test_update_elements_3d_data() {
        let device = CudaDevice::new(0).unwrap();

        let data_shape = vec![2, 2, 2];
        let mut data_stride = vec![0; data_shape.len()];
        utils::calculate_stride(&data_shape, &mut data_stride);
        let data = vec![0.0; data_shape.iter().product()];

        let indices_shape = vec![2, 3];
        let mut indices_stride = vec![0; indices_shape.len()];
        utils::calculate_stride(&indices_shape, &mut indices_stride);
        let indices = device.htod_copy(vec![1, 0, 1, 0, 1, 0]).unwrap();

        let updates_shape = vec![2];
        let mut updates_stride = vec![0; updates_shape.len()];
        utils::calculate_stride(&updates_shape, &mut updates_stride);
        let updates = device.htod_copy(vec![9.0, 7.0]).unwrap();

        let output_shape = data_shape.clone();
        let mut output_stride = vec![0; output_shape.len()];
        utils::calculate_stride(&output_shape, &mut output_stride);
        let mut output = device.htod_copy(data).unwrap();

        let f = load_kernel(&device.clone(), ScatterNdKernel::FwdF32).unwrap();

        let info = create_info_buffer(
            &data_shape,
            &data_stride,
            &indices_shape,
            &indices_stride,
            &updates_shape,
            &updates_stride,
        );

        let mut error = device.alloc_zeros(1).unwrap();

        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let updates_rank = updates_shape.len();

        let num_idx_tuples = indices_shape[0..indices_rank - 1].iter().product();

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                num_idx_tuples,
                data_rank,
                indices_rank,
                updates_rank,
                &info,
                &indices,
                &updates,
                &mut output,
                &mut error,
            )
            .unwrap();
        }

        let result = device.dtoh_sync_copy(&output).unwrap();

        assert_eq!(result, vec![0.0, 0.0, 7.0, 0.0, 0.0, 9.0, 0.0, 0.0])
    }

    #[test]
    fn test_update_slices_3d_data() {
        let device = CudaDevice::new(0).unwrap();

        let data_shape = vec![4, 4, 3];
        let mut data_stride = vec![0; data_shape.len()];
        utils::calculate_stride(&data_shape, &mut data_stride);
        let data = vec![0.0; data_shape.iter().product()];

        let indices_shape = vec![2, 2];
        let mut indices_stride = vec![0; indices_shape.len()];
        utils::calculate_stride(&indices_shape, &mut indices_stride);
        let indices = device.htod_copy(vec![0, 1, 2, 2]).unwrap();

        let updates_shape = vec![2, 3];
        let mut updates_stride = vec![0; updates_shape.len()];
        utils::calculate_stride(&updates_shape, &mut updates_stride);
        let updates = device
            .htod_copy(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        let output_shape = data_shape.clone();
        let mut output_stride = vec![0; output_shape.len()];
        utils::calculate_stride(&output_shape, &mut output_stride);
        let mut output = device.htod_copy(data).unwrap();

        let f = load_kernel(&device.clone(), ScatterNdKernel::FwdF32).unwrap();

        let info = create_info_buffer(
            &data_shape,
            &data_stride,
            &indices_shape,
            &indices_stride,
            &updates_shape,
            &updates_stride,
        );

        let mut error = device.alloc_zeros(1).unwrap();

        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let updates_rank = updates_shape.len();

        let num_idx_tuples = indices_shape[0..indices_rank - 1].iter().product();

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                num_idx_tuples,
                data_rank,
                indices_rank,
                updates_rank,
                &info,
                &indices,
                &updates,
                &mut output,
                &mut error,
            )
            .unwrap();
        }

        let error = device.dtoh_sync_copy(&error).unwrap();
        assert_eq!(error[0], 0);

        let result = device.dtoh_sync_copy(&output).unwrap();

        assert_eq!(
            result,
            vec![
                0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0,
                6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        )
    }

    #[test]
    fn test_update_elements_1d_indices() {
        let device = CudaDevice::new(0).unwrap();

        let data_shape = vec![5];
        let mut data_stride = vec![0; data_shape.len()];
        utils::calculate_stride(&data_shape, &mut data_stride);
        let data = vec![0.0; data_shape.iter().product()];

        let indices_shape = vec![2];
        let mut indices_stride = vec![0; indices_shape.len()];
        utils::calculate_stride(&indices_shape, &mut indices_stride);
        let indices = device.htod_copy(vec![1, 3]).unwrap();

        let updates_shape = vec![2];
        let mut updates_stride = vec![0; updates_shape.len()];
        utils::calculate_stride(&updates_shape, &mut updates_stride);
        let updates = device.htod_copy(vec![10.0, 20.0]).unwrap();

        let output_shape = data_shape.clone();
        let mut output_stride = vec![0; output_shape.len()];
        utils::calculate_stride(&output_shape, &mut output_stride);
        let mut output = device.htod_copy(data).unwrap();

        let f = load_kernel(&device.clone(), ScatterNdKernel::FwdF32).unwrap();

        let info = create_info_buffer(
            &data_shape,
            &data_stride,
            &indices_shape,
            &indices_stride,
            &updates_shape,
            &updates_stride,
        );

        let mut error = device.alloc_zeros(1).unwrap();

        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let updates_rank = updates_shape.len();

        // Because its rank=1, it's implied K=1;
        let num_idx_tuples = indices_shape.iter().product();

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                num_idx_tuples,
                data_rank,
                indices_rank,
                updates_rank,
                &info,
                &indices,
                &updates,
                &mut output,
                &mut error,
            )
            .unwrap();
        }

        let error = device.dtoh_sync_copy(&error).unwrap();
        assert_eq!(error[0], 0);

        let result = device.dtoh_sync_copy(&output).unwrap();

        assert_eq!(result, vec![0.0, 10.0, 0.0, 20.0, 0.0])
    }

    #[test]
    fn test_update_slices_1d_indices() {
        let device = CudaDevice::new(0).unwrap();

        let data_shape = vec![4, 3];
        let mut data_stride = vec![0; data_shape.len()];
        utils::calculate_stride(&data_shape, &mut data_stride);
        let data = vec![0.0; data_shape.iter().product()];

        let indices_shape = vec![2];
        let mut indices_stride = vec![0; indices_shape.len()];
        utils::calculate_stride(&indices_shape, &mut indices_stride);
        let indices = device.htod_copy(vec![1, 3]).unwrap();

        let updates_shape = vec![2, 3];
        let mut updates_stride = vec![0; updates_shape.len()];
        utils::calculate_stride(&updates_shape, &mut updates_stride);
        let updates = device
            .htod_copy(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        let output_shape = data_shape.clone();
        let mut output_stride = vec![0; output_shape.len()];
        utils::calculate_stride(&output_shape, &mut output_stride);
        let mut output = device.htod_copy(data).unwrap();

        let f = load_kernel(&device.clone(), ScatterNdKernel::FwdF32).unwrap();

        let info = create_info_buffer(
            &data_shape,
            &data_stride,
            &indices_shape,
            &indices_stride,
            &updates_shape,
            &updates_stride,
        );

        let mut error = device.alloc_zeros(1).unwrap();

        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let updates_rank = updates_shape.len();

        // Because its rank=1, it's implied K=1;
        let num_idx_tuples = indices_shape.iter().product();

        unsafe {
            compute::<f32>(
                device.clone(),
                f,
                num_idx_tuples,
                data_rank,
                indices_rank,
                updates_rank,
                &info,
                &indices,
                &updates,
                &mut output,
                &mut error,
            )
            .unwrap();
        }

        let error = device.dtoh_sync_copy(&error).unwrap();
        assert_eq!(error[0], 0);

        let result = device.dtoh_sync_copy(&output).unwrap();

        assert_eq!(
            result,
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0]
        )
    }
}
