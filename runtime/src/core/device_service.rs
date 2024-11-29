use crate::core::error::Result;
use crate::core::kernel::Kernel;
use rmlk_schema::{DataType, Op};

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

    fn alloc_zeros_float(&self, len: usize) -> Result<Self::Data>;
}
