use crate::core::error::UnsupportedDataType;
use crate::core::Context;
use crate::providers::cuda;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::utils;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::{debug, trace};
use num_traits::Num;
use rmlk_cuda::kernels::whereop;
use rmlk_cuda::kernels::whereop::WhereKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::collections::HashMap;
use std::sync::Arc;

pub struct WhereBackend {
    stream: Arc<CudaStream>,
}

impl WhereBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl WhereBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => WhereKernel::WhereFwdF16,
            DataType::Float => WhereKernel::WhereFwdF32,
            DataType::Double => WhereKernel::WhereFwdF64,
            DataType::Int32 => WhereKernel::WhereFwdI32,
            DataType::Int64 => WhereKernel::WhereFwdI64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        whereop::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_where<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(T::data_type())?;

        compute_output_shape(ctx)?;

        let condition = ctx.get_input(0)?;
        let x = ctx.get_input(1)?;
        let y = ctx.get_input(2)?;

        let x_dev_data_ref = x.payload();
        let x_dev_data = x_dev_data_ref.data::<T>();

        let y_dev_data_ref = y.payload();
        let y_dev_data = y_dev_data_ref.data::<T>();

        let condition_dev_data_ref = condition.payload();
        let condition_dev_data = condition_dev_data_ref.data::<bool>();

        let output = ctx.get_output(0)?;

        if condition.is_scalar() && x.is_scalar() && y.is_scalar() {
            output.init_scalar_payload::<T>()?;
        } else {
            output.init_payload::<T>()?;
        }

        debug!(
            "[output][dtype={:?}][where][shape={:?}][stride=[{:?}]",
            output.dtype(),
            output.shape(),
            output.stride()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let (x_shape, x_stride) = if x.is_scalar() {
            cuda::utils::scalar_shape_and_stride(&x)
        } else {
            (x.shape(), x.stride())
        };

        let output_shape = if output.is_scalar() {
            let (shape, _) = cuda::utils::scalar_shape_and_stride(&output);
            shape
        } else {
            output.shape()
        };

        let broadcast_x_stride = scratch_alloc.allocate_fill::<usize>(output_shape.len(), 0)?;
        utils::compute_broadcast_stride_from_output_shape(
            &x_shape,
            &x_stride,
            &output_shape,
            broadcast_x_stride,
        );

        let (y_shape, y_stride) = if y.is_scalar() {
            cuda::utils::scalar_shape_and_stride(&y)
        } else {
            (y.shape(), y.stride())
        };

        let broadcast_y_stride = scratch_alloc.allocate_fill::<usize>(output_shape.len(), 0)?;
        utils::compute_broadcast_stride_from_output_shape(
            &y_shape,
            &y_stride,
            &output_shape,
            broadcast_y_stride,
        );

        let (condition_shape, condition_stride) = if condition.is_scalar() {
            cuda::utils::scalar_shape_and_stride(&condition)
        } else {
            (condition.shape(), condition.stride())
        };

        let broadcast_condition_stride =
            scratch_alloc.allocate_fill::<usize>(output_shape.len(), 0)?;
        utils::compute_broadcast_stride_from_output_shape(
            &condition_shape,
            &condition_stride,
            &output_shape,
            broadcast_condition_stride,
        );

        debug!("[x][where][broadcast][stride={:?}]", broadcast_x_stride);
        debug!("[y][where][broadcast][stride={:?}]", broadcast_y_stride);
        debug!(
            "[condition][where][broadcast][stride={:?}]",
            broadcast_condition_stride
        );

        let output_rank = output_shape.len();

        let info_on_host = ctx
            .execution_state()
            .scratch_alloc()
            .allocate(4 * output_rank)?;
        info_on_host[..output_rank].copy_from_slice(&output_shape);
        info_on_host[output_rank..2 * output_rank].copy_from_slice(broadcast_x_stride);
        info_on_host[2 * output_rank..3 * output_rank].copy_from_slice(broadcast_y_stride);
        info_on_host[3 * output_rank..].copy_from_slice(broadcast_condition_stride);

        trace!("[info_on_host={:?}]", info_on_host);

        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let info = cuda_bump.alloc_from_slice_with_fallback(info_on_host)?;
        let info_data = info.data::<usize>();

        {
            let mut output_payload = output.payload_mut();
            let mut output_data = output_payload.data_mut();
            unsafe {
                whereop::compute(
                    self.stream.clone(),
                    func,
                    output_rank,
                    &info_data,
                    &x_dev_data,
                    &y_dev_data,
                    &condition_dev_data,
                    &mut output_data,
                )?;
            }
        }

        #[cfg(feature = "dump")]
        debug::write_results_ternary::<bool, T, T, T>(
            "debugging/where",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(1)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_where::<f16>(ctx),
            DataType::Float => self.compute_where::<f32>(ctx),
            DataType::Double => self.compute_where::<f64>(ctx),
            DataType::Int32 => self.compute_where::<i32>(ctx),
            DataType::Int64 => self.compute_where::<i64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn compute_output_shape(ctx: &mut Context<Cuda>) -> Result<()> {
    let condition = ctx.get_input(0)?;

    debug!(
        "[condition][dtype={:?}][shape={:?}][stride={:?}]",
        condition.dtype(),
        condition.shape(),
        condition.stride()
    );

    let x = ctx.get_input(1)?;

    debug!(
        "[x][dtype={:?}][shape={:?}][stride={:?}]",
        x.dtype(),
        x.shape(),
        x.stride()
    );

    let y = ctx.get_input(2)?;

    debug!(
        "[y][dtype={:?}][shape={:?}][stride={:?}]",
        y.dtype(),
        y.shape(),
        y.stride()
    );

    match x.shape().as_ref() == y.shape().as_ref()
        && x.shape().as_ref() == condition.shape().as_ref()
    {
        true => {
            let output = ctx.get_output(0)?;
            output.copy_shape(x.shape_handle());
        }
        false => {
            let rank = [x.shape().len(), y.shape().len(), condition.shape().len()]
                .into_iter()
                .max()
                .expect("Iterator is not empty");

            let alloc = ctx.execution_state().scratch_alloc().clone();
            let inter_shape = alloc.allocate_fill(rank, 0)?;

            if !utils::compute_broadcast_output_shape(&x.shape(), &y.shape(), inter_shape) {
                return Err(WhereError::IncompatibleShapesForBroadcast {
                    shapes: [(1, x.shape().to_vec()), (2, y.shape().to_vec())].into(),
                }
                .into());
            }

            let output_shape = alloc.allocate_fill(rank, 0)?;

            if !utils::compute_broadcast_output_shape(inter_shape, &condition.shape(), output_shape)
            {
                return Err(WhereError::IncompatibleShapesForBroadcast {
                    shapes: [(0, y.shape().to_vec())].into(),
                }
                .into());
            }

            let output = ctx.get_output(0)?;
            output.copy_shape_from_slice(output_shape);
        }
    }

    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum WhereError {
    #[error("incompatible shapes for broadcast: {shapes:?}")]
    IncompatibleShapesForBroadcast { shapes: HashMap<usize, Vec<usize>> },
}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::Op;

    #[test]
    fn two_inputs() {
        let out = OpTest::new(Op::Where)
            .input([2, 2], vec![true, false, true, false])
            .input([2, 2], vec![11.0f32, 22.0, 33.0, 44.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .output([2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![11.0, 2.0, 33.0, 4.0]);
    }

    #[test]
    fn all_scalars() {
        let out = OpTest::new(Op::Where)
            .input([], vec![true])
            .input([], vec![11.1f32])
            .input([], vec![1.0f32])
            .output([])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![11.1]);
    }

    #[test]
    fn one_scalar() {
        let out = OpTest::new(Op::Where)
            .input([2, 1], vec![true, false])
            .input([1, 2], vec![1.0f32, 2.0])
            .input([], vec![0.0f32])
            .output([2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn x_y_scalars() {
        let out = OpTest::new(Op::Where)
            .input([3], vec![true, false, true])
            .input([], vec![11.1f32])
            .input([], vec![1.0f32])
            .output([3])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![11.1, 1.0, 11.1]);
    }

    #[test]
    fn broadcast_diff_shape_len() {
        let out = OpTest::new(Op::Where)
            .constant(
                [2, 6],
                vec![
                    false, true, true, false, true, true, false, false, false, false, true, true,
                ],
            )
            .input(
                [3, 2, 1],
                vec![
                    0.40744543f32,
                    0.09136569,
                    0.72850689,
                    0.33760798,
                    0.802_622_4,
                    0.41559307,
                ],
            )
            .input(
                [1, 6],
                vec![
                    0.698_947_4f32,
                    0.53463184,
                    0.809_403_2,
                    0.864_580_3,
                    0.34860543,
                    0.67579115,
                ],
            )
            .output([3, 2, 6])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                0.698_947_4,
                0.40744543,
                0.40744543,
                0.864_580_3,
                0.40744543,
                0.40744543,
                0.698_947_4,
                0.53463184,
                0.809_403_2,
                0.864_580_3,
                0.09136569,
                0.09136569,
                0.698_947_4,
                0.72850689,
                0.72850689,
                0.864_580_3,
                0.72850689,
                0.72850689,
                0.698_947_4,
                0.53463184,
                0.809_403_2,
                0.864_580_3,
                0.33760798,
                0.33760798,
                0.698_947_4,
                0.802_622_4,
                0.802_622_4,
                0.864_580_3,
                0.802_622_4,
                0.802_622_4,
                0.698_947_4,
                0.53463184,
                0.809_403_2,
                0.864_580_3,
                0.41559307,
                0.41559307,
            ]
        );
    }
}
