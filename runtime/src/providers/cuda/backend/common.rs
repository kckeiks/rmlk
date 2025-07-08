use crate::core::Context;
use crate::providers::cuda::Cuda;
use anyhow::Result;

pub fn unary_op_copy_shape(ctx: &Context<Cuda>) -> Result<()> {
    let input = ctx.get_input(0)?;
    let output = ctx.get_output(0)?;
    output.copy_shape(input.shape_handle());
    Ok(())
}
