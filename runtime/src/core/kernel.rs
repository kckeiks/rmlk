use crate::core::context::Context;
use crate::core::error::Result;
use crate::core::DeviceService;

pub trait Kernel {
    type Device: DeviceService;
    fn compute(self, ctx: &mut Context<Self::Device>) -> Result<()>;
}
