use crate::core::error::UnsupportedDataType;
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::expand;
use rmlk_cuda::kernels::expand::ExpandKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ExpandBackend {
    stream: Arc<CudaStream>,
}

impl ExpandBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => ExpandKernel::FwdF16,
            DataType::Float => ExpandKernel::FwdF32,
            DataType::Double => ExpandKernel::FwdF64,
            DataType::Int32 => ExpandKernel::FwdI32,
            DataType::Uint32 => ExpandKernel::FwdU32,
            DataType::Int64 => ExpandKernel::FwdI64,
            DataType::Uint64 => ExpandKernel::FwdU64,
            _ => {
                return Err(UnsupportedDataType(dtype).into());
            }
        };

        debug!("[kernel={:?}]", kernel_name);

        expand::load_kernel(self.stream.context(), kernel_name).map_err(Into::into)
    }

    fn compute_expand<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(T::data_type())?;

        compute_output_shape(ctx)?;

        let output_tensor = ctx.get_output(0)?;
        output_tensor.init_payload::<T>()?;

        debug!(
            "[output][dtype={:?}][shape={:?}][stride={:?}]",
            output_tensor.dtype(),
            output_tensor.shape(),
            output_tensor.stride()
        );

        let input_tensor = ctx.get_input(0)?;
        let input_payload = input_tensor.payload();
        let input_data = input_payload.data::<T>();

        let input_rank = if input_tensor.is_scalar() {
            1
        } else {
            input_tensor.shape().len()
        };

        let output_rank = output_tensor.shape().len();
        let elem_count = output_tensor.shape().iter().product();

        let info = create_info(ctx)?;
        let info_data = info.data::<usize>();

        {
            let mut output_payload = output_tensor.payload_mut();
            let mut output_data = output_payload.data_mut::<T>();

            unsafe {
                expand::compute(
                    self.stream.clone(),
                    func,
                    input_rank,
                    output_rank,
                    &info_data,
                    elem_count,
                    input_data.as_ref(),
                    output_data.as_mut(),
                )?;
            }
        }

        #[cfg(feature = "dump")]
        debug::write_results_binary::<T, i64, T>(
            "debugging/expand",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_expand::<f16>(ctx),
            DataType::Float => self.compute_expand::<f32>(ctx),
            DataType::Double => self.compute_expand::<f64>(ctx),
            DataType::Int32 => self.compute_expand::<i32>(ctx),
            DataType::Uint32 => self.compute_expand::<u32>(ctx),
            DataType::Int64 => self.compute_expand::<i64>(ctx),
            DataType::Uint64 => self.compute_expand::<u64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn compute_output_shape(ctx: &Context<Cuda>) -> Result<()> {
    let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

    let input_tensor = ctx.get_input(0)?;

    debug!(
        "[input][dtype={:?}][shape={:?}][stride={:?}]",
        input_tensor.dtype(),
        input_tensor.shape(),
        input_tensor.stride()
    );

    let shape_tensor = ctx.get_input(1)?;

    debug!(
        "[shape][dtype=i64][shape={:?}][stride={:?}]",
        shape_tensor.shape(),
        shape_tensor.stride()
    );

    let rank = shape_tensor.shape().iter().product();

    let shape_payload = shape_tensor.payload();
    let shape_data = shape_payload.data::<i64>();
    let shape_on_host = scratch_alloc.allocate::<i64>(shape_data.len())?;
    shape_tensor.payload_to_host(shape_on_host)?;

    let target_shape =
        scratch_alloc.allocate_and_convert_from_slice::<i64, usize>(shape_on_host)?;

    let output_shape = scratch_alloc.allocate(rank)?;

    if !utils::compute_broadcast_output_shape(&input_tensor.shape(), target_shape, output_shape) {
        return Err(ExpandError::IncompatibleShapesForBroadcast {
            shapes: [
                (0, input_tensor.shape().to_vec()),
                (1, shape_tensor.shape().to_vec()),
            ]
            .into(),
        }
        .into());
    }

    let output_tensor = ctx.get_output(0)?;
    output_tensor.copy_shape_from_slice(output_shape);

    Ok(())
}

fn create_info(ctx: &Context<Cuda>) -> Result<CudaData> {
    let input_tensor = ctx.get_input(0)?;
    let output_tensor = ctx.get_output(0)?;

    let input_rank = if input_tensor.is_scalar() {
        1
    } else {
        input_tensor.shape().len()
    };
    let output_rank = output_tensor.shape().len();

    let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
    let info = match (input_tensor.is_scalar(), output_tensor.is_scalar()) {
        (false, false) => scratch_alloc.allocate(2 * input_rank + 2 * output_rank)?,
        (true, true) => scratch_alloc.allocate(4)?,
        (true, false) => scratch_alloc.allocate(2 + 2 * output_rank)?,
        (false, true) => scratch_alloc.allocate(2 + 2 * input_rank)?,
    };

    if input_tensor.is_scalar() {
        utils::write_info(&[1], &[1], info, 0);
    } else {
        utils::write_info(&input_tensor.shape(), &input_tensor.stride(), info, 0);
    }

    let start = if input_tensor.is_scalar() {
        2
    } else {
        2 * input_rank
    };

    if output_tensor.is_scalar() {
        utils::write_info(&[1], &[1], info, start);
    } else {
        utils::write_info(&output_tensor.shape(), &output_tensor.stride(), info, start);
    }

    let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
    cuda_bump.alloc_from_slice_with_fallback(info)
}

#[derive(Debug, thiserror::Error)]
pub enum ExpandError {
    #[error("incompatible shapes for broadcast: {shapes:?}")]
    IncompatibleShapesForBroadcast { shapes: HashMap<usize, Vec<usize>> },
}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::Op;

    #[test]
    fn basic() {
        let out = OpTest::new(Op::Expand)
            .input([2, 1, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .input([3], vec![2i64, 4, 3])
            .output([2, 4, 3])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0,
                5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
            ]
        );
    }

    #[test]
    fn scalar_input() {
        // Input is a true scalar; expand uses init paths that tolerate empty input shape.
        let out = OpTest::new(Op::Expand)
            .input([], vec![5.0f32])
            .input([2], vec![2i64, 3])
            .output([2, 3])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![5.0; 6]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Expand)
            .input([1, 1], vec![true])
            .input([2], vec![1i64, 1])
            .output([1, 1])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
