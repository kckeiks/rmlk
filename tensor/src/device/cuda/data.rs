use crate::device::cuda;
use cudarc::driver::CudaSlice;
use half::f16;

#[derive(Clone)]
pub enum Data {
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl Data {
    pub fn f16(&self) -> cuda::Result<&CudaSlice<f16>> {
        match &self {
            Self::F16(slice) => Ok(slice),
            _ => return Err(()),
        }
    }

    pub fn f32(&self) -> cuda::Result<&CudaSlice<f32>> {
        match &self {
            Self::F32(slice) => Ok(slice),
            _ => return Err(()),
        }
    }

    pub fn f64(&self) -> cuda::Result<&CudaSlice<f64>> {
        match &self {
            Self::F64(slice) => Ok(slice),
            _ => return Err(()),
        }
    }
}
