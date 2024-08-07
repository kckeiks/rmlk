use crate::core::kernel::Kernel;
use crate::Result;
use rmlk_ir::{DataType, Op};

pub trait DeviceService {
    /// Data on device.
    type Data;
    type Kernel: Kernel<Device = Self>;

    fn get_kernel(&self, op: Op, dtype: DataType) -> Result<Self::Kernel>;

    fn htod_float(&self, data: Vec<f32>) -> Result<Self::Data>;

    fn dtoh_float(&self, data: &mut Self::Data) -> Result<Vec<f32>>;
}
