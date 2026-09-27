use crate::ptx::SLICE;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use std::sync::Arc;

pub const PTX_SRC: &str = SLICE;

#[derive(Debug)]
pub enum SliceKernel {
    FwdF16WithIndexI32,
    FwdF16WithIndexI64,
    FwdF32WithIndexI32,
    FwdF32WithIndexI64,
    FwdF64WithIndexI32,
    FwdF64WithIndexI64,
    FwdI32WithIndexI32,
    FwdI32WithIndexI64,
    FwdI64WithIndexI32,
    FwdI64WithIndexI64,
}

impl From<SliceKernel> for &'static str {
    fn from(value: SliceKernel) -> Self {
        match value {
            SliceKernel::FwdF16WithIndexI32 => "slice_fwd_f16_with_index_i32",
            SliceKernel::FwdF16WithIndexI64 => "slice_fwd_f16_with_index_i64",
            SliceKernel::FwdF32WithIndexI32 => "slice_fwd_f32_with_index_i32",
            SliceKernel::FwdF32WithIndexI64 => "slice_fwd_f32_with_index_i64",
            SliceKernel::FwdF64WithIndexI32 => "slice_fwd_f64_with_index_i32",
            SliceKernel::FwdF64WithIndexI64 => "slice_fwd_f64_with_index_i64",
            SliceKernel::FwdI32WithIndexI32 => "slice_fwd_i32_with_index_i32",
            SliceKernel::FwdI32WithIndexI64 => "slice_fwd_i32_with_index_i64",
            SliceKernel::FwdI64WithIndexI32 => "slice_fwd_i64_with_index_i32",
            SliceKernel::FwdI64WithIndexI64 => "slice_fwd_i64_with_index_i64",
        }
    }
}

pub fn load_kernel(
    ctx: Arc<CudaContext>,
    kernel_name: SliceKernel,
) -> crate::error::Result<CudaFunction> {
    let module = ctx.load_module(PTX_SRC.into())?;
    module.load_function(kernel_name.into()).map_err(Into::into)
}

// Assumptions for this CUDA Slice kernel:
//
// 1. `rank <= 8` and validated on the host.
// 2. `indices_len == starts.len() == ends.len() == axes.len() == steps.len()`.
// 3. `axes[k] ∈ [–rank, rank)` for all k.
// 4. `steps[k] != 0` for all k.
// 5. `num_elems == output_shape.iter().product()`.
// 6. `output_shape[d] <= TYPENAME_INDEX::MAX` for each d.
// 7. `input_shape`, `input_stride`, `output_shape`, `output_stride` are valid arrays of length `4*rank`.
// 8. `output` tensor is C-contiguous (so writing `output[i]` is valid).
// 9. Host precomputes `output_shape[d] = ceil((end–start)/step)` per ONNX rules.
// 10. The host enforces `num_elems <= SIZE_MAX` and grid/block dimensions do not overflow `size_t`.
pub unsafe fn compute<T, Tind>(
    stream: Arc<CudaStream>,
    func: CudaFunction,
    rank: usize,
    info: &CudaSlice<usize>,
    starts: &CudaSlice<Tind>,
    ends: &CudaSlice<Tind>,
    axes: &CudaSlice<Tind>,
    steps: &CudaSlice<Tind>,
    input_data: &CudaSlice<T>,
    output_data: &mut CudaSlice<T>,
) -> crate::error::Result<()>
where
    T: ValidAsZeroBits + DeviceRepr,
    Tind: ValidAsZeroBits + DeviceRepr,
{
    let elem_count = output_data.len();

    assert_eq!(info.len(), 4 * rank);

    if elem_count == 0 {
        return Ok(());
    }

    // Todo: we need to validate this.
    // assert_eq!(elem_count, info[2 * rank..3 * rank].iter().product());

    assert!(8 >= rank);

    let axes_len = axes.len();
    let indices_len = starts.len();
    assert_eq!(ends.len(), indices_len);
    assert!(axes_len == indices_len || axes_len == 0);
    assert!(steps.len() == indices_len || steps.is_empty());

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
            .arg(&axes_len)
            .arg(&rank)
            .arg(info)
            .arg(input_data)
            .arg(starts)
            .arg(ends)
            .arg(axes)
            .arg(steps)
            .arg(output_data)
            .launch(config)?;
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils;

    fn launch_slice_test<T, Tind>(
        input: &[T],
        input_shape: &[usize],
        output_shape: &[usize],
        starts: &[Tind],
        ends: &[Tind],
        axes: &[Tind],
        steps: &[Tind],
        kernel_type: SliceKernel,
    ) -> Vec<T>
    where
        T: ValidAsZeroBits + DeviceRepr + Clone + Unpin + PartialEq + Default,
        Tind: ValidAsZeroBits + DeviceRepr + Clone + Unpin + PartialEq + Default,
    {
        let rank = input_shape.len();

        let mut in_strides = vec![0; rank];
        if !input_shape.is_empty() {
            utils::calculate_stride(input_shape, &mut in_strides);
        }
        let mut out_strides = vec![0; rank];
        if !output_shape.is_empty() {
            utils::calculate_stride(output_shape, &mut out_strides);
        }

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let func = load_kernel(ctx, kernel_type).unwrap();

        let mut info = Vec::with_capacity(rank * 4);
        info.extend_from_slice(input_shape);
        info.extend_from_slice(&in_strides);
        info.extend_from_slice(output_shape);
        info.extend_from_slice(&out_strides);
        let info = stream.clone_htod(&info).unwrap();

        let starts_dev = stream.clone_htod(starts).unwrap();
        let ends_dev = stream.clone_htod(ends).unwrap();
        let axes_dev = if axes.is_empty() {
            stream.null().unwrap()
        } else {
            stream.clone_htod(axes).unwrap()
        };
        let steps_dev = if steps.is_empty() {
            stream.null().unwrap()
        } else {
            stream.clone_htod(steps).unwrap()
        };
        let input_dev = stream.clone_htod(input).unwrap();
        let elem_count = output_shape.iter().product::<usize>();
        let mut d_output = stream.alloc_zeros::<T>(elem_count).unwrap();

        unsafe {
            compute(
                stream.clone(),
                func,
                rank,
                &info,
                &starts_dev,
                &ends_dev,
                &axes_dev,
                &steps_dev,
                &input_dev,
                &mut d_output,
            )
            .unwrap();
        }

        stream.clone_dtoh(&d_output).unwrap()
    }

    #[test]
    fn test_slice_scalar() {
        let input: [f32; 1] = [42.0];

        let input_shape: &[usize] = &[];
        let output_shape: &[usize] = &[];
        let starts: &[i64] = &[];
        let ends: &[i64] = &[];
        let axes: &[i64] = &[];
        let steps: &[i64] = &[];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, input);
    }

    #[test]
    fn test_slice_identity_1d() {
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let input_shape: &[usize] = &[5];
        let output_shape: &[usize] = &[5];
        let starts: &[i64] = &[0];
        let ends: &[i64] = &[5];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, input);
    }

    #[test]
    fn test_slice_1d_basic() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected: Vec<f32> = vec![2.0, 3.0, 4.0];

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[3];
        let starts: &[i64] = &[2];
        let ends: &[i64] = &[5];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_slice_2d_axis1() {
        let input: Vec<f32> = vec![
            10.0, 20.0, 30.0, 40.0, // row 0
            50.0, 60.0, 70.0, 80.0, // row 1
        ];
        let expected = vec![20.0, 30.0, 60.0, 70.0];

        let input_shape: &[usize] = &[2, 4];
        let output_shape: &[usize] = &[2, 2];
        let starts: &[i64] = &[1];
        let ends: &[i64] = &[3];
        let axes: &[i64] = &[1];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_slice_step_and_negative() {
        let input: Vec<f32> = (0..8).map(|n| n as f32).collect();
        let expected = vec![6.0, 4.0, 2.0, 0.0];

        let input_shape: &[usize] = &[8];
        let output_shape: &[usize] = &[4];
        let starts: &[i64] = &[6];
        let ends: &[i64] = &[i64::MIN]; // ONNX INT_MIN sentinel for “to beginning”
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[-2];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_rank1_step1() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[6];
        let starts: &[i64] = &[2];
        let ends: &[i64] = &[8];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_rank1_step2_neg() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected = vec![7.0, 5.0, 3.0];

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[3];
        let starts: &[i64] = &[7];
        let ends: &[i64] = &[1];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[-2];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_rank2_mixed_steps() {
        let input: Vec<f32> = (0..12).map(|n| n as f32).collect();
        let expected = vec![5.0, 7.0, 9.0, 11.0];

        let input_shape: &[usize] = &[3, 4];
        let output_shape: &[usize] = &[2, 2];
        let starts: &[i64] = &[1, 1];
        let ends: &[i64] = &[3, 4];
        let axes: &[i64] = &[0, 1];
        let steps: &[i64] = &[1, 2];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_rank3_slice_last_axis() {
        let input: Vec<f32> = (0..12).map(|n| n as f32).collect();
        let expected = vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0];

        let input_shape: &[usize] = &[2, 2, 3];
        let output_shape: &[usize] = &[2, 2, 2];
        let starts: &[i64] = &[1];
        let ends: &[i64] = &[3];
        let axes: &[i64] = &[2];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_negative_start_clamping() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected = vec![5.0, 6.0, 7.0];

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[3];
        let starts: &[i64] = &[-5];
        let ends: &[i64] = &[8];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_negative_end_clamping_positive() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[6];
        let starts: &[i64] = &[2];
        let ends: &[i64] = &[-2];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_overlarge_end_positive() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected: Vec<f32> = input.clone();

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[10];
        let starts: &[i64] = &[0];
        let ends: &[i64] = &[20];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_overlarge_start_positive() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected: Vec<f32> = vec![];

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[0];
        let starts: &[i64] = &[20];
        let ends: &[i64] = &[25];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_overlarge_start_negative_step() {
        let input: Vec<f32> = (0..5).map(|n| n as f32).collect();
        let expected = vec![4.0];

        let input_shape: &[usize] = &[5];
        let output_shape: &[usize] = &[1];
        let starts: &[i64] = &[10];
        let ends: &[i64] = &[-1];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[-1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_too_negative_end_negative_step() {
        let input: Vec<f32> = (0..6).map(|n| n as f32).collect();
        let expected = vec![4.0, 2.0, 0.0];

        let input_shape: &[usize] = &[6];
        let output_shape: &[usize] = &[3];
        let starts: &[i64] = &[4];
        let ends: &[i64] = &[-10];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[-2];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_stride_positive_step_gt1() {
        let input: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let expected = vec![0.0, 2.0, 4.0, 6.0, 8.0];

        let input_shape: &[usize] = &[10];
        let output_shape: &[usize] = &[5];
        let starts: &[i64] = &[0];
        let ends: &[i64] = &[10];
        let axes: &[i64] = &[0];
        let steps: &[i64] = &[2];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_negative_indexed_axes() {
        let input: Vec<f32> = (0..6).map(|n| n as f32).collect();
        let expected = vec![1.0, 2.0, 4.0, 5.0];

        let input_shape: &[usize] = &[2, 3];
        let output_shape: &[usize] = &[2, 2];
        let starts: &[i64] = &[1];
        let ends: &[i64] = &[3];
        let axes: &[i64] = &[-1];
        let steps: &[i64] = &[1];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_null_axes_and_steps() {
        let input: Vec<f32> = (0..12).map(|n| n as f32).collect();
        let expected = vec![5.0, 6.0, 9.0, 10.0];

        let input_shape: &[usize] = &[3, 4];
        let output_shape: &[usize] = &[2, 2];
        let starts: &[i64] = &[1, 1];
        let ends: &[i64] = &[3, 4];
        let axes: &[i64] = &[];
        let steps: &[i64] = &[];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_null_axes() {
        let input: Vec<f32> = (0..12).map(|n| n as f32).collect();
        let expected = vec![5.0, 7.0, 9.0, 11.0];

        let input_shape: &[usize] = &[3, 4];
        let output_shape: &[usize] = &[2, 2];
        let starts: &[i64] = &[1, 1];
        let ends: &[i64] = &[3, 4];
        let axes: &[i64] = &[];
        let steps: &[i64] = &[1, 2];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_null_steps() {
        let input: Vec<f32> = (0..12).map(|n| n as f32).collect();
        let expected = vec![5.0, 6.0, 9.0, 10.0];

        let input_shape: &[usize] = &[3, 4];
        let output_shape: &[usize] = &[2, 2];
        let starts: &[i64] = &[1, 1];
        let ends: &[i64] = &[3, 4];
        let axes: &[i64] = &[0, 1];
        let steps: &[i64] = &[];

        let out = launch_slice_test::<f32, i64>(
            &input,
            input_shape,
            output_shape,
            starts,
            ends,
            axes,
            steps,
            SliceKernel::FwdF32WithIndexI64,
        );
        assert_eq!(out, expected);
    }
}
