use crate::ptx::SCATTER_ND;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
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

// Todo: Add description about cases where this will panic.
/// Launches a CUDA kernel that performs an ScatterND operation.
///
/// # Safety
/// - The `func` CUDA function must accept the following arguments in that order:
///   - The number of index tuples.
///   - The rank of the data.
///   - The rank of the indices.
///   - The rank of the updates.
///   - The `info_buffer` must be structured as:
///      - The **stride of `data`**.
///      - The **shape of `indices`**.
///      - The **stride of `indices`**.
///      - The **shape of `updates`**.
///      - The **stride of `updates`**.
///   - The input data slice `indices`.
///   - The input data slice `updates`.
///   - The output data slice `output`.
///   - The data slice `error` of length 1.
///
/// Note that this function assumes that the output slice equals to the data in the `data` input.
pub unsafe fn compute<T>(
    stream: Arc<CudaStream>,
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
    let info = stream.memcpy_stod(&info[data_rank..])?;

    let num_threads = 128;
    let num_blocks = (num_idx_tuples + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&num_idx_tuples)
            .arg(&output.len())
            .arg(&data_rank)
            .arg(&indices_rank)
            .arg(&updates_rank)
            .arg(&info)
            .arg(indices)
            .arg(updates)
            .arg(output)
            .arg(error)
            .launch(config)?;
    }

    Ok(())
}

pub fn load_kernel(
    ctx: &Arc<CudaContext>,
    kernel_name: ScatterNdKernel,
) -> crate::error::Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::scatter_nd::{compute, load_kernel, ScatterNdKernel};
    use crate::utils;
    use cudarc::driver::CudaContext;

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

    fn run_scatter_nd_test(
        data_shape: Vec<usize>,
        indices_shape: Vec<usize>,
        indices_data: Vec<usize>,
        updates_shape: Vec<usize>,
        updates_data: Vec<f32>,
        expected_output: Vec<f32>,
    ) {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let mut data_stride = vec![0; data_shape.len()];
        utils::calculate_stride(&data_shape, &mut data_stride);
        let data = vec![0.0; data_shape.iter().product()];

        let mut indices_stride = vec![0; indices_shape.len()];
        utils::calculate_stride(&indices_shape, &mut indices_stride);
        let indices = stream.memcpy_stod(&indices_data).unwrap();

        let mut updates_stride = vec![0; updates_shape.len()];
        utils::calculate_stride(&updates_shape, &mut updates_stride);
        let updates = stream.memcpy_stod(&updates_data).unwrap();

        let mut output_stride = vec![0; data_shape.len()];
        utils::calculate_stride(&data_shape, &mut output_stride);
        let mut output = stream.memcpy_stod(&data).unwrap();

        let f = load_kernel(&ctx, ScatterNdKernel::FwdF32).unwrap();

        let info = create_info_buffer(
            &data_shape,
            &data_stride,
            &indices_shape,
            &indices_stride,
            &updates_shape,
            &updates_stride,
        );

        let mut error = stream.alloc_zeros(1).unwrap();

        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let updates_rank = updates_shape.len();

        let num_idx_tuples = if indices_rank == 1 {
            indices_shape.iter().product()
        } else {
            indices_shape[0..indices_rank - 1].iter().product()
        };

        unsafe {
            compute::<f32>(
                stream.clone(),
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

        let error_data = stream.memcpy_dtov(&error).unwrap();
        assert_eq!(error_data[0], 0);

        let result = stream.memcpy_dtov(&output).unwrap();
        assert_eq!(result, expected_output);
    }

    #[test]
    fn test_update_elements_2d_data() {
        let data_shape = vec![3, 3];
        let indices_shape = vec![2, 2];
        let indices_data = vec![0, 1, 2, 2];
        let updates_shape = vec![2];
        let updates_data = vec![5.0, 8.0];
        let expected_output = vec![0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0];

        run_scatter_nd_test(
            data_shape,
            indices_shape,
            indices_data,
            updates_shape,
            updates_data,
            expected_output,
        );
    }

    #[test]
    fn test_update_elements_3d_data() {
        let data_shape = vec![2, 2, 2];
        let indices_shape = vec![2, 3];
        let indices_data = vec![1, 0, 1, 0, 1, 0];
        let updates_shape = vec![2];
        let updates_data = vec![9.0, 7.0];
        let expected_output = vec![0.0, 0.0, 7.0, 0.0, 0.0, 9.0, 0.0, 0.0];

        run_scatter_nd_test(
            data_shape,
            indices_shape,
            indices_data,
            updates_shape,
            updates_data,
            expected_output,
        );
    }

    #[test]
    fn test_update_slices_3d_data() {
        let data_shape = vec![4, 4, 3];
        let indices_shape = vec![2, 2];
        let indices_data = vec![0, 1, 2, 2];
        let updates_shape = vec![2, 3];
        let updates_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected_output = vec![
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        run_scatter_nd_test(
            data_shape,
            indices_shape,
            indices_data,
            updates_shape,
            updates_data,
            expected_output,
        );
    }

    #[test]
    fn test_update_elements_1d_indices() {
        let data_shape = vec![5];
        let indices_shape = vec![2];
        let indices_data = vec![1, 3];
        let updates_shape = vec![2];
        let updates_data = vec![10.0, 20.0];
        let expected_output = vec![0.0, 10.0, 0.0, 20.0, 0.0];

        run_scatter_nd_test(
            data_shape,
            indices_shape,
            indices_data,
            updates_shape,
            updates_data,
            expected_output,
        );
    }

    #[test]
    fn test_update_slices_1d_indices() {
        let data_shape = vec![4, 3];
        let indices_shape = vec![2];
        let indices_data = vec![1, 3];
        let updates_shape = vec![2, 3];
        let updates_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected_output = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0];

        run_scatter_nd_test(
            data_shape,
            indices_shape,
            indices_data,
            updates_shape,
            updates_data,
            expected_output,
        );
    }
}
