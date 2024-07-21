use crate::{Error, Result};
use cudarc::driver::CudaSlice;
use half::f16;

#[derive(Clone)]
pub enum CudaData {
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl CudaData {
    pub fn f16(&self) -> Result<&CudaSlice<f16>> {
        match &self {
            Self::F16(slice) => Ok(slice),
            _ => return Err(Error::Unknown),
        }
    }

    pub fn f32(&self) -> Result<&CudaSlice<f32>> {
        match &self {
            Self::F32(slice) => Ok(slice),
            _ => return Err(Error::Unknown),
        }
    }

    pub fn f32_mut(&mut self) -> Result<&mut CudaSlice<f32>> {
        match self {
            Self::F32(slice) => Ok(slice),
            _ => return Err(Error::Unknown),
        }
    }

    pub fn f64(&self) -> Result<&CudaSlice<f64>> {
        match &self {
            Self::F64(slice) => Ok(slice),
            _ => return Err(Error::Unknown),
        }
    }
}
