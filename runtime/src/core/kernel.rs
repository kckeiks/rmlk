use crate::core::context::Context;
use crate::core::error::Result;
use crate::core::ExecutionProvider;

pub trait Kernel {
    type Provider: ExecutionProvider;
    fn compute(self, ctx: &mut Context<Self::Provider>) -> Result<()>;
}
