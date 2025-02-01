use cudarc::cublas::result::CublasError;
use cudarc::cudnn::CudnnError;
use cudarc::driver::DriverError;

#[derive(Debug)]
pub enum Error {
    NonContiguousMemory(String),
    InvalidArguments(String),
    Internal(String),
    Cuda(u32),
    Cublas(u32),
    Cudnn(u32),
}

impl From<CublasError> for Error {
    fn from(value: CublasError) -> Self {
        Self::Cublas(value.0 as u32)
    }
}

impl From<CudnnError> for Error {
    fn from(value: CudnnError) -> Self {
        Self::Cudnn(value.0 as u32)
    }
}

impl From<DriverError> for Error {
    fn from(value: DriverError) -> Self {
        Self::Cuda(value.0 as u32)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
