use crate::core::backend::OperationBackend;
use crate::core::error::Result;
use rmlk_schema::{DataType, Op};

/// Services for using an accelerator device's resources.
pub trait DeviceService: Sized {
    /// Data on device.
    type Data: DeviceData;
    /// Backends that executes kernels on device.
    type Backend: OperationBackend<Self>;

    /// Get the backend for an operation.
    fn get_backend(&self, op: Op, dtype: DataType) -> Result<Self::Backend>;

    /// Copies `f32` data from host to device.
    fn htod_float(&self, data: Vec<f32>) -> Result<Self::Data>;

    /// Copies `f32` data from device to host.
    fn dtoh_float(&self, data: &Self::Data) -> Result<Vec<f32>>;
    fn dtoh_i64(&self, data: &Self::Data) -> Result<Vec<i64>>;

    /// Copies `i32` data from host to device.
    fn htod_i32(&self, data: Vec<i32>) -> Result<Self::Data>;

    /// Copies `i32` data from host to device.
    fn htod_i64(&self, data: Vec<i64>) -> Result<Self::Data>;
    fn alloc_zeros_float(&self, len: usize) -> Result<Self::Data>;
}

pub trait DeviceData {
    fn dtype(&self) -> DataType;
}
