use crate::context::ExecutionContext;
use crate::provider::Provider;

pub trait Kernel {
    type Provider: Provider;
    fn compute(&self, ctx: &mut ExecutionContext<Self::Provider>) -> crate::Result<()>;
}
