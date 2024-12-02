use crate::core::error::Result;
use crate::core::kernel::KernelBackend;
use rmlk_schema::{DataType, Op};

/// Services for using an accelerator device's resources.
pub trait DeviceService {
    /// Data on device.
    type Data;
    /// Kernel that can be executed on device.
    type Backend: KernelBackend<Device = Self>;

    /// Get the kernel backend given the operation and data type.
    fn get_kernel_backend(&self, op: Op, dtype: DataType) -> Result<Self::Backend>;

    /// Copies `f32` data from host to device.
    fn htod_float(&self, data: Vec<f32>) -> Result<Self::Data>;

    /// Copies `f32` data from device to host.
    fn dtoh_float(&self, data: &Self::Data) -> Result<Vec<f32>>;

    fn alloc_zeros_float(&self, len: usize) -> Result<Self::Data>;
}
