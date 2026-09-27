use crate::attributes::error::AttributeError;
use crate::core::error::UnsupportedDataType;
use crate::core::Context;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use crate::{attributes, utils};
use anyhow::Result;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct ConcatBackend {
    stream: Arc<CudaStream>,
}

impl ConcatBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }

    fn run_computation<T>(&self, ctx: &Context<Cuda>, axis: usize) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let output_tensor = ctx.get_output(0)?;
        let mut output_payload = output_tensor.payload_mut();
        let mut output_data = output_payload.data_mut::<T>();

        let input = ctx.get_input(0)?;

        let outer_dims = input.shape()[..axis].iter().product::<usize>();
        let mut axis_offset = 0;
        for i in 0..usize::MAX {
            match ctx.get_input(i) {
                Ok(input) => {
                    let dim = input.shape()[axis];
                    let step = input.stride()[axis];
                    let outer_block_size = step * dim;
                    let output_outer_block_size = output_tensor.shape()[axis] * step;

                    let input_payload = input.payload();
                    let input_data = input_payload.data::<T>();
                    for outer_i in 0..outer_dims {
                        for j in 0..dim {
                            let output_offset =
                                (outer_i * output_outer_block_size) + (axis_offset + j) * step;

                            self.stream.memcpy_dtod(
                                &input_data.slice(
                                    (outer_i * outer_block_size) + (j * step)
                                        ..(outer_i * outer_block_size) + (j * step) + step,
                                ),
                                &mut output_data.slice_mut(output_offset..output_offset + step),
                            )?;
                        }
                    }
                    axis_offset += dim;
                }
                _ => break,
            }
        }

        assert_eq!(
            output_data.len(),
            output_tensor.shape().iter().product::<usize>()
        );

        Ok(())
    }

    // TODO: If axis is last dimension (step == 1), consider larger DtoD copies
    fn compute_concat<T>(&mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let input_tensor = ctx.get_input(0)?;

        // Todo: I think the `compute_output_shape` will validate the rest of the tensors.
        // We may not need this.
        if input_tensor.is_scalar() {
            return Err(ConcatError::ScalarsAreNotAllowed.into());
        }

        let attrs = ctx
            .get_attributes()
            .ok_or(AttributeError::MissingAttributes)
            .map_err(Box::new)?;

        // Todo: validate axis.
        let raw_axis = attributes::concat::get_axis(&attrs).ok_or_else(|| {
            AttributeError::MissingAttribute {
                name: "`axis` is missing".to_string(),
            }
        })?;

        debug!("[attributes][axis={:?}]", raw_axis);

        let axis = utils::normalize_index(raw_axis as i64, input_tensor.shape().len())?;

        compute_output_shape(ctx, axis)?;

        {
            let output_tensor = ctx.get_output(0)?;
            output_tensor.init_payload::<T>()?;

            debug!(
                "[output][dtype={:?}][shape={:?}][stride=[{:?}]",
                output_tensor.dtype(),
                output_tensor.shape(),
                output_tensor.stride()
            );

            self.run_computation::<T>(ctx, axis)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_concat::<T>("debugging/concat", self.stream.clone(), ctx)?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_concat::<f16>(ctx),
            DataType::Float => self.compute_concat::<f32>(ctx),
            DataType::Double => self.compute_concat::<f64>(ctx),
            DataType::Int32 => self.compute_concat::<i32>(ctx),
            DataType::Uint32 => self.compute_concat::<u32>(ctx),
            DataType::Int64 => self.compute_concat::<i64>(ctx),
            DataType::Uint64 => self.compute_concat::<u64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn compute_output_shape(ctx: &Context<Cuda>, target_axis: usize) -> Result<()> {
    let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

    let input_tensor = ctx.get_input(0)?;
    let base_shape = scratch_alloc.allocate_from_slice(&input_tensor.shape())?;

    let mut shape_on_axis = 0;
    for i in 0..usize::MAX {
        match ctx.get_input(i) {
            Ok(next_input_tensor) => {
                debug!(
                    "[input][{i}][dtype={:?}][shape={:?}][stride=[{:?}]",
                    next_input_tensor.dtype(),
                    next_input_tensor.shape(),
                    next_input_tensor.stride()
                );

                if next_input_tensor.shape().len() != base_shape.len() {
                    return Err(ConcatError::ShapeMismatch {
                        index: i,
                        base_rank: base_shape.len(),
                        found_rank: next_input_tensor.shape().len(),
                    }
                    .into());
                }

                shape_on_axis += next_input_tensor.shape()[target_axis];

                for (axis, dim) in next_input_tensor.shape().iter().enumerate() {
                    if axis == target_axis {
                        continue;
                    }

                    if *dim != base_shape[axis] {
                        return Err(ConcatError::DimensionMismatch {
                            axis,
                            dim: *dim,
                            expected_dim: base_shape[axis],
                        }
                        .into());
                    }
                }
            }
            _ => break,
        }
    }

    base_shape[target_axis] = shape_on_axis;

    let output = ctx.get_output(0)?;
    output.copy_shape_from_slice(base_shape);

    Ok(())
}

#[derive(Debug)]
pub enum ConcatError {
    ShapeMismatch {
        index: usize,
        base_rank: usize,
        found_rank: usize,
    },
    DimensionMismatch {
        axis: usize,
        dim: usize,
        expected_dim: usize,
    },
    ScalarsAreNotAllowed,
}

impl Display for ConcatError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConcatError::ShapeMismatch { index, base_rank, found_rank } => write!(f, "shape mismatch for input {index}: base rank {base_rank} and found rank {found_rank}"),
            ConcatError::DimensionMismatch { axis, dim, expected_dim } => write!(f, "dimension mismatch for input {axis} and input dim {dim} expected dim {expected_dim}"),
            ConcatError::ScalarsAreNotAllowed => write!(f, "scalars are not allowed"),
        }
    }
}

impl std::error::Error for ConcatError {}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::{AttributeType, Op};

    #[test]
    fn rank2_axis0() {
        let out = OpTest::new(Op::Concat)
            .input([3, 2], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .input([2, 2], vec![100.0f32, 101.0, 200.0, 201.0])
            .attr("axis", AttributeType::Int(0))
            .output([5, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 101.0, 200.0, 201.0]
        );
    }

    #[test]
    fn rank2_axis1() {
        let out = OpTest::new(Op::Concat)
            .input([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .input([2, 2], vec![100.0f32, 101.0, 200.0, 201.0])
            .attr("axis", AttributeType::Int(1))
            .output([2, 5])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![1.0, 2.0, 3.0, 100.0, 101.0, 4.0, 5.0, 6.0, 200.0, 201.0]
        );
    }

    #[test]
    fn rank3_axis1() {
        let a: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let b: Vec<f32> = (100..=107).map(|x| x as f32).collect();
        let out = OpTest::new(Op::Concat)
            .input([2, 3, 4], a)
            .input([2, 1, 4], b)
            .attr("axis", AttributeType::Int(1))
            .output([2, 4, 4])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 100.0, 101.0, 102.0,
                103.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                104.0, 105.0, 106.0, 107.0,
            ]
        );
    }

    #[test]
    fn rank3_axis2() {
        let a: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let b: Vec<f32> = (100..=111).map(|x| x as f32).collect();
        let out = OpTest::new(Op::Concat)
            .input([2, 3, 4], a)
            .input([2, 3, 2], b)
            .attr("axis", AttributeType::Int(2))
            .output([2, 3, 6])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 5.0, 6.0, 7.0, 8.0, 102.0, 103.0, 9.0, 10.0,
                11.0, 12.0, 104.0, 105.0, 13.0, 14.0, 15.0, 16.0, 106.0, 107.0, 17.0, 18.0, 19.0,
                20.0, 108.0, 109.0, 21.0, 22.0, 23.0, 24.0, 110.0, 111.0,
            ]
        );
    }

    #[test]
    fn rank4_axis2() {
        let a: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let b = vec![
            101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0,
            201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0,
        ];
        let out = OpTest::new(Op::Concat)
            .input([2, 3, 2, 4], a)
            .input([2, 3, 1, 4], b)
            .attr("axis", AttributeType::Int(2))
            .output([2, 3, 3, 4])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 101.0, 102.0, 103.0, 104.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 105.0, 106.0, 107.0, 108.0, 17.0, 18.0, 19.0,
                20.0, 21.0, 22.0, 23.0, 24.0, 109.0, 110.0, 111.0, 112.0, 25.0, 26.0, 27.0, 28.0,
                29.0, 30.0, 31.0, 32.0, 201.0, 202.0, 203.0, 204.0, 33.0, 34.0, 35.0, 36.0, 37.0,
                38.0, 39.0, 40.0, 205.0, 206.0, 207.0, 208.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
                47.0, 48.0, 209.0, 210.0, 211.0, 212.0,
            ]
        );
    }

    #[test]
    fn rank4_axis3() {
        let a: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let b = vec![
            101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0,
        ];
        let out = OpTest::new(Op::Concat)
            .input([2, 3, 2, 4], a)
            .input([2, 3, 2, 1], b)
            .attr("axis", AttributeType::Int(3))
            .output([2, 3, 2, 5])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 3.0, 4.0, 101.0, 5.0, 6.0, 7.0, 8.0, 102.0, 9.0, 10.0, 11.0, 12.0, 103.0,
                13.0, 14.0, 15.0, 16.0, 104.0, 17.0, 18.0, 19.0, 20.0, 105.0, 21.0, 22.0, 23.0,
                24.0, 106.0, 25.0, 26.0, 27.0, 28.0, 107.0, 29.0, 30.0, 31.0, 32.0, 108.0, 33.0,
                34.0, 35.0, 36.0, 109.0, 37.0, 38.0, 39.0, 40.0, 110.0, 41.0, 42.0, 43.0, 44.0,
                111.0, 45.0, 46.0, 47.0, 48.0, 112.0,
            ]
        );
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Concat)
            .input([1, 1], vec![true])
            .input([1, 1], vec![false])
            .attr("axis", AttributeType::Int(0))
            .output([2, 1])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
