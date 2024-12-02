use crate::core::context::Context;
use crate::core::device_service::DeviceService;
use crate::core::error::Result;

pub trait KernelBackend {
    type Device: DeviceService;

    fn compute(self, ctx: &mut Context<Self::Device>) -> Result<()>;
}
