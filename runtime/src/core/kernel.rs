use crate::core::context::Context;
use crate::core::device_service::DeviceService;

pub type Result<T> = std::result::Result<T, KernelError>;

#[derive(Debug)]
pub enum KernelError {
    InvalidTensorDimensions(Vec<usize>),
    MissingAttributes,
    Other(String),
}

pub trait Kernel {
    type Device: DeviceService;
    fn compute(self, ctx: &mut Context<Self::Device>) -> Result<()>;
}
