use crate::core::device_service::DeviceServiceError;
use crate::core::kernel::KernelError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Device(DeviceServiceError),
    Internal(String),
    Kernel(KernelError),
    ModelDeserializationFailed,
}

impl From<KernelError> for Error {
    fn from(value: KernelError) -> Self {
        Self::Kernel(value)
    }
}

impl From<DeviceServiceError> for Error {
    fn from(value: DeviceServiceError) -> Self {
        Self::Device(value)
    }
}
