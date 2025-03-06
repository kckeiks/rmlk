use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::Cuda;
use cudarc::driver::CudaDevice;
use rmlk_schema::{DataType, Op};
use std::sync::Arc;
use crate::attributes::transpose;

pub struct TransposeBackend {
    device: Arc<CudaDevice>,
}

impl TransposeBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    fn compute_transpose<I>(&mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let input = ctx.get_input(0)?;

        let alloc = ctx.execution_state().scratch_alloc().clone();

        let output_shape = alloc.allocate(input.shape().len())?;

        match ctx.get_attributes().and_then(transpose::get_perm) {
            None => {}
            Some(perm) => {
                // The length of perm must be equal to the rank of the input.
                if perm.len() != output_shape.len() {
                    
                }
                
                for dim_i in perm {
                    output_shape[*dim_i] = input.shape()[dim_i];
                }
            }
        }
        let dst = ctx.get_output(0)?.dst_id();

        ctx.execution_state_mut()
            .copy_shape_from_slice(output_shape, dst)?;

        Ok(())
    }

    pub fn compute(mut self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_transpose::<f32>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Transpose,
                dtype,
            }),
        }
    }
}
