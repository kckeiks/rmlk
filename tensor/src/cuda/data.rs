use cudarc::driver::CudaSlice;
use half::f16;

#[derive(Clone)]
pub enum CudaData {
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl CudaData {
    pub fn f16(&self) -> Option<&CudaSlice<f16>> {
        match &self {
            Self::F16(slice) => Some(slice),
            _ => None,
        }
    }

    pub fn f32(&self) -> Option<&CudaSlice<f32>> {
        match &self {
            Self::F32(slice) => Some(slice),
            _ => None,
        }
    }

    pub fn f32_mut(&mut self) -> Option<&mut CudaSlice<f32>> {
        match self {
            Self::F32(slice) => Some(slice),
            _ => None,
        }
    }

    pub fn f64(&self) -> Option<&CudaSlice<f64>> {
        match &self {
            Self::F64(slice) => Some(slice),
            _ => None,
        }
    }
}
