use crate::core::kernel::Kernel;
use rmlk_schema::{DataType, Op};

pub type Result<T> = std::result::Result<T, DeviceServiceError>;

#[derive(Debug)]
pub enum DeviceServiceError {
    Cuda(rmlk_cuda::Error),
    Other(String),
}

impl From<rmlk_cuda::Error> for DeviceServiceError {
    fn from(value: rmlk_cuda::Error) -> Self {
        DeviceServiceError::Cuda(value)
    }
}

/// Services for using an accelerator device's resources.
pub trait DeviceService {
    /// Data on device.
    type Data;
    /// Kernel that can be executed on device.
    type Kernel: Kernel<Device = Self>;

    /// Get the kernel given the operation and data type.
    fn get_kernel(&self, op: Op, dtype: DataType) -> Result<Self::Kernel>;

    /// Copies `f32` data from host to device.
    fn htod_float(&self, data: Vec<f32>) -> Result<Self::Data>;

    /// Copies `f32` data from device to host.
    fn dtoh_float(&self, data: &Self::Data) -> Result<Vec<f32>>;
}
