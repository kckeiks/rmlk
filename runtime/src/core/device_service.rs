use crate::core::backend::OperationBackend;
use anyhow::Result;
use cudarc::driver::DeviceRepr;
use half::f16;
use rmlk_graph::Graph;
use rmlk_schema::{DataType, DataTypeMap, Definition, Op};
use std::collections::HashMap;
use std::fmt::Debug;

/// Services for using an accelerator device's resources.
pub trait DeviceService: Sized {
    /// Backends that executes kernels on device.
    type Backend: OperationBackend<Self>;
    type Value: Value;

    type Store: ValueStore<Value = Self::Value>;

    /// Get the backend for an operation.
    fn get_backend(&self, op: Op, dtype: DataType) -> Result<Self::Backend>;
    fn store(&self) -> Result<Self::Store>;
    /// Copies `f32` data from host to device.
    fn htod_float16(&self, data: Vec<f16>) -> Result<<Self::Value as Value>::Data>;
    fn htod_float(&self, data: Vec<f32>) -> Result<<Self::Value as Value>::Data>;
    fn htod_double(&self, data: Vec<f64>) -> Result<<Self::Value as Value>::Data>;
    /// Copies `f32` data from device to host.
    fn dtoh_float16(&self, data: &<Self::Value as Value>::Data) -> Result<Vec<f16>>;
    fn dtoh_float(&self, data: &<Self::Value as Value>::Data) -> Result<Vec<f32>>;
    fn dtoh_i32(&self, data: &<Self::Value as Value>::Data) -> Result<Vec<i32>>;
    fn dtoh_i64(&self, data: &<Self::Value as Value>::Data) -> Result<Vec<i64>>;
    fn dtoh_bool(&self, data: &<Self::Value as Value>::Data) -> Result<Vec<bool>>;
    /// Copies `i32` data from host to device.
    fn htod_i32(&self, data: Vec<i32>) -> Result<<Self::Value as Value>::Data>;
    /// Copies `i32` data from host to device.
    fn htod_i64(&self, data: Vec<i64>) -> Result<<Self::Value as Value>::Data>;
    fn htod_bool(&self, data: Vec<bool>) -> Result<<Self::Value as Value>::Data>;
    fn alloc_zeros_float(&self, len: usize) -> Result<<Self::Value as Value>::Data>;
}

pub trait DeviceData: Debug {}

pub trait ValueStore {
    type Value: Value;

    fn init(
        &mut self,
        graph: &Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Result<()>;
    fn get(&self, id: usize) -> Option<Self::Value>;
    fn clear(&self);
}

pub trait Value {
    type Data: DeviceData;
    fn set_data(&mut self, data: Self::Data) -> Result<()>;
    fn set_shape(&mut self, shape: &[usize]) -> Result<()>;
    fn data<T>(&self) -> Result<Vec<T>>
    where
        T: DataTypeMap + DeviceRepr + Default + Clone;
    fn shape(&self) -> Vec<usize>;
    fn dtype(&self) -> DataType;
}
