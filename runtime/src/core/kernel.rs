use crate::core::allocator::AllocatorError;
use crate::core::context::Context;
use crate::core::device_service::{DeviceService, DeviceServiceError};
pub type Result<T> = std::result::Result<T, KernelError>;

#[derive(Debug)]
pub enum KernelError {
    Device(DeviceServiceError),
    InvalidTensorDimensions(Vec<usize>),
    MissingAttributes,
    HostAllocator(AllocatorError),
    Other(String),
}

pub trait Kernel {
    type Device: DeviceService;
    fn compute(self, ctx: &mut Context<Self::Device>) -> Result<()>;
}

impl From<DeviceServiceError> for KernelError {
    fn from(value: DeviceServiceError) -> Self {
        Self::Device(value)
    }
}

impl From<AllocatorError> for KernelError {
    fn from(value: AllocatorError) -> Self {
        Self::HostAllocator(value)
    }
}
