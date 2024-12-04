use crate::core::context::Context;
use crate::core::device_service::DeviceService;
use crate::core::error::Result;

pub trait OperationBackend<T: DeviceService> {
    /// Computes the operation for this backend.
    fn compute(self, ctx: &mut Context<T>) -> Result<()>;
}
