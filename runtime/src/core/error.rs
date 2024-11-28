use crate::core::device_service::DeviceServiceError;
use crate::core::kernel::KernelError;
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, InternalError>;

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

impl From<InternalError> for Error {
    fn from(value: InternalError) -> Self {
        Self::Internal(value.to_string())
    }
}

#[derive(Debug)]
pub enum InternalError {
    TensorStore(String),
    ExecutionState(String),
    Device(DeviceServiceError),
}

impl Display for InternalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalError::TensorStore(msg) => {
                write!(f, "tensor store error: {msg}")
            }
            InternalError::ExecutionState(msg) => {
                write!(f, "execution state error: {msg}")
            }
            InternalError::Device(msg) => {
                write!(f, "device error: {msg:?}")
            }
        }
    }
}

impl From<DeviceServiceError> for InternalError {
    fn from(value: DeviceServiceError) -> Self {
        Self::Device(value)
    }
}
