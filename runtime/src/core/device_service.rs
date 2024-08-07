use crate::core::kernel::Kernel;
use crate::core::tensor::Tensor;
use crate::Result;
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Op};
use std::collections::HashMap;

pub trait DeviceService {
    /// Data on device.
    type Data;
    type Kernel: Kernel<Device = Self>;
    fn allocate_execution_state(
        &mut self,
        graph: &Graph,
        plan: &[usize],
    ) -> Result<(
        HashMap<usize, usize>,
        Box<[Option<Tensor<Self::Data>>]>,
        Box<[usize]>,
    )>;
    fn get_kernel(&self, op: Op, dtype: DataType) -> Result<Self::Kernel>;

    fn htod_float(&self, data: Vec<f32>) -> Result<Self::Data>;

    fn dtoh_float(&self, data: &mut Self::Data) -> Result<Vec<f32>>;
}
