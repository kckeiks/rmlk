use crate::core::error::UnsupportedDataType;
use crate::core::Context;

use crate::providers::cuda::Cuda;
use crate::{attributes, utils};
use anyhow::Result;
use cudarc::driver::CudaStream;
use log::debug;
use num_traits::ToPrimitive;
use rmlk_schema::DataType;
use std::sync::Arc;

pub struct ShapeBackend {
    _stream: Arc<CudaStream>,
}

impl ShapeBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            _stream: stream.clone(),
        }
    }

    pub fn compute_shape(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let data = ctx.get_input(0)?;

        debug!(
            "[data][dtype={:?}][shape={:?}][strides={:?}]",
            data.dtype(),
            data.shape(),
            data.stride()
        );

        let rank = data.shape().len();

        let attrs = ctx.get_attributes();

        debug!("[attributes={attrs:?}]");

        let raw_start = attrs
            .as_ref()
            .map(|attrs| attributes::shape::get_start(attrs.as_ref()))
            .unwrap_or(0);
        let raw_end = match attrs.and_then(|attrs| attributes::shape::get_end(&attrs)) {
            None => rank.to_i32().ok_or(ShapeError::UnsupportedRankSize {
                message: format!("failed to convert `{rank}` to i32"),
            })?,
            Some(end) => end,
        };

        let (start, end) = utils::derive_range(raw_start as i64, raw_end as i64, rank)?;

        compute_output_shape(start, end, ctx)?;

        let output = ctx.get_output(0)?;
        output.init_payload::<i64>()?;

        debug!(
            "[output][dtype={:?}][shape={:?}][strides={:?}]",
            output.dtype(),
            output.shape(),
            output.stride()
        );

        let shape = ctx.get_output(0)?;

        let shape_host_buf = ctx
            .execution_state()
            .scratch_alloc()
            .allocate_and_convert_from_slice::<_, i64>(&data.shape())?;

        shape.write_payload_from_slice(&shape_host_buf[start..end])?;

        #[cfg(feature = "dump")]
        write_shape_dump(self._stream.clone(), ctx)?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16
            | DataType::Float
            | DataType::Double
            | DataType::Int32
            | DataType::Int64 => self.compute_shape(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

#[cfg(feature = "dump")]
fn write_shape_dump(stream: Arc<CudaStream>, ctx: &mut Context<Cuda>) -> Result<()> {
    use crate::providers::cuda::debug;
    use half::f16;

    match ctx.get_input(0)?.dtype() {
        DataType::Float16 => debug::write_results_shape::<f16>("debugging/shape", stream, ctx),
        DataType::Float => debug::write_results_shape::<f32>("debugging/shape", stream, ctx),
        DataType::Double => debug::write_results_shape::<f64>("debugging/shape", stream, ctx),
        DataType::Int32 => debug::write_results_shape::<i32>("debugging/shape", stream, ctx),
        DataType::Int64 => debug::write_results_shape::<i64>("debugging/shape", stream, ctx),
        dtype => Err(UnsupportedDataType(dtype).into()),
    }
}

pub fn compute_output_shape(start: usize, end: usize, ctx: &Context<Cuda>) -> Result<()> {
    let alloc = ctx.execution_state().scratch_alloc().clone();
    let tensor_shape_buf = alloc.allocate_fill::<usize>(1, end - start)?;
    let tensor = ctx.get_output(0)?;
    tensor.copy_shape_from_slice(tensor_shape_buf);
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    #[error("unsupported rank size: {message}")]
    UnsupportedRankSize { message: String },
}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::{AttributeType, Op};

    #[test]
    fn simple() {
        let out = OpTest::new(Op::Shape)
            .input([3, 4, 2], vec![0.0f32; 3 * 4 * 2])
            .output([3])
            .run::<i64>()
            .unwrap();
        assert_eq!(out, vec![3, 4, 2]);
    }

    #[test]
    fn scalar() {
        let out = OpTest::new(Op::Shape)
            .input([], vec![3.3f32])
            .output([0])
            .run::<i64>()
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn start_end() {
        let out = OpTest::new(Op::Shape)
            .input([3, 4, 2], vec![0.0f32; 3 * 4 * 2])
            .attr("start", AttributeType::Int(0))
            .attr("end", AttributeType::Int(2))
            .output([2])
            .run::<i64>()
            .unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn negative_start_end() {
        let out = OpTest::new(Op::Shape)
            .input([3, 4, 2], vec![0.0f32; 3 * 4 * 2])
            .attr("start", AttributeType::Int(-2))
            .attr("end", AttributeType::Int(-1))
            .output([1])
            .run::<i64>()
            .unwrap();
        assert_eq!(out, vec![4]);
    }

    #[test]
    fn negative_start_only() {
        let out = OpTest::new(Op::Shape)
            .input([3, 4, 2], vec![0.0f32; 3 * 4 * 2])
            .attr("start", AttributeType::Int(-2))
            .output([2])
            .run::<i64>()
            .unwrap();
        assert_eq!(out, vec![4, 2]);
    }

    #[test]
    fn negative_end_only() {
        let out = OpTest::new(Op::Shape)
            .input([3, 4, 2], vec![0.0f32; 3 * 4 * 2])
            .attr("end", AttributeType::Int(-1))
            .output([2])
            .run::<i64>()
            .unwrap();
        assert_eq!(out, vec![3, 4]);
    }
}
