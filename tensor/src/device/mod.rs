use half::f16;
use std::alloc::Allocator;

pub mod cuda;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Unknown,
}

pub trait Device {
    type Data: Clone;
    type Cpu: Allocator + Clone;

    fn htod_f16(&self, data: Vec<f16>) -> Result<Self::Data>;
    fn htod_f32(&self, data: Vec<f32>) -> Result<Self::Data>;
    fn htod_f64(&self, data: Vec<f64>) -> Result<Self::Data>;
    fn dtoh_f16(&self, data: &Self::Data) -> Result<Vec<f16>>;
    fn dtoh_f32(&self, data: &Self::Data) -> Result<Vec<f32>>;
    fn dtoh_f64(&self, data: &Self::Data) -> Result<Vec<f64>>;
}
