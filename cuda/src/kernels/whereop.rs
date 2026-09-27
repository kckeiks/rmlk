use crate::error::Result;
use crate::ptx::WHERE;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const MODULE_NAME: &str = "where";
pub const FWD_FN_NAMES: [&str; 3] = ["where_fwd_f16", "where_fwd_f32", "where_fwd_f64"];
pub const PTX_SRC: &str = WHERE;

pub enum WhereKernel {
    WhereFwdF16,
    WhereFwdF32,
    WhereFwdF64,
    WhereFwdI32,
    WhereFwdI64,
}

impl From<WhereKernel> for &'static str {
    fn from(value: WhereKernel) -> Self {
        match value {
            WhereKernel::WhereFwdF16 => "where_fwd_f16",
            WhereKernel::WhereFwdF32 => "where_fwd_f32",
            WhereKernel::WhereFwdF64 => "where_fwd_f64",
            WhereKernel::WhereFwdI32 => "where_fwd_i32",
            WhereKernel::WhereFwdI64 => "where_fwd_i64",
        }
    }
}

pub fn load_kernel(ctx: Arc<CudaContext>, kernel_name: WhereKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

/// Launches a CUDA kernel that performs an element-wise conditional selection (`where` operation).
///
/// This function applies the following operation:
/// ```text
/// output[i] = if z[i] != 0 { x[i] } else { y[i] }
/// ```
/// Supports **multidirectional (NumPy-style) broadcasting** for inputs of different shapes.
///
/// # Safety
/// - The `info_buffer` **must contain exactly `4 * ndims` elements**, structured as:
///   - First `ndims` entries: **Output shape**.
///   - Next `ndims` entries: **Strides for `x`**.
///   - Next `ndims` entries: **Strides for `y`**.
///   - Last `ndims` entries: **Strides for `z`**.
/// - Input tensors (`x_data`, `y_data`, `z_data`) **must be allocated on the CUDA device** and match their corresponding shapes and strides.
///
/// # Panics
/// - Panics if info buffer does not equal to 4 * `ndims`.
/// - Panics if output slice does not have the expected size based on the output shape.
pub unsafe fn compute<T>(
    stream: Arc<CudaStream>,
    func: CudaFunction,
    ndims: usize,
    info: &CudaSlice<usize>,
    x_data: &CudaSlice<T>,
    y_data: &CudaSlice<T>,
    z_data: &CudaSlice<bool>,
    output_data: &mut CudaSlice<T>,
) -> Result<()>
where
    T: ValidAsZeroBits + DeviceRepr,
{
    assert_eq!(4 * ndims, info.len());

    let elem_count = output_data.len();

    // Todo: we need to validate that shape is valid and that it's consistent with
    // the length of the cuda slice.
    //assert_eq!(elem_count, output_data.len());

    let num_threads = 128;
    let num_blocks = elem_count.div_ceil(num_threads);

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(&elem_count)
            .arg(&ndims)
            .arg(info)
            .arg(x_data)
            .arg(y_data)
            .arg(z_data)
            .arg(output_data)
            .launch(config)?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::kernels::whereop::{compute, load_kernel, WhereKernel};
    use crate::utils;
    use cudarc::driver::CudaContext;

    fn create_info_buffer(
        output_shape: &[usize],
        x_stride: &[usize],
        y_stride: &[usize],
        z_stride: &[usize],
    ) -> Vec<usize> {
        let ndims = output_shape.len();
        let mut info_buffer = vec![0usize; 4 * ndims];

        info_buffer[..ndims].copy_from_slice(output_shape);
        info_buffer[ndims..2 * ndims].copy_from_slice(x_stride);
        info_buffer[2 * ndims..3 * ndims].copy_from_slice(y_stride);
        info_buffer[3 * ndims..].copy_from_slice(z_stride);

        info_buffer
    }

    #[test]
    fn test_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let x_shape = vec![2, 2];
        let mut x_stride = vec![0; x_shape.len()];
        utils::calculate_stride(&x_shape, &mut x_stride);
        let x_data = stream.clone_htod(&vec![10.0, 20.0, 30.0, 40.0]).unwrap();

        let y_shape = vec![2, 2];
        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);
        let y_data = stream.clone_htod(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let z_shape = vec![2, 2];
        let mut z_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&z_shape, &mut z_stride);
        let z_data = stream.clone_htod(&vec![true, false, false, true]).unwrap();

        let f = load_kernel(ctx.clone(), WhereKernel::WhereFwdF32).unwrap();

        let output_shape = vec![2, 2];
        let mut out_data = stream
            .alloc_zeros(output_shape.iter().copied().product())
            .unwrap();

        let info = create_info_buffer(&output_shape, &x_stride, &y_stride, &z_stride);
        let info = stream.clone_htod(&info).unwrap();

        unsafe {
            compute::<f32>(
                stream.clone(),
                f,
                output_shape.len(),
                &info,
                &x_data,
                &y_data,
                &z_data,
                &mut out_data,
            )
            .unwrap();
        }

        let result = stream.clone_dtoh(&out_data).unwrap();

        assert_eq!(result, vec![10.0, 2.0, 3.0, 40.0])
    }
}
