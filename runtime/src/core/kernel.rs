use crate::core::context::Context;
use crate::core::error::Result;

pub trait Kernel {
    type Data;
    fn compute(self, ctx: &mut Context<Self::Data>) -> Result<()>;
}
