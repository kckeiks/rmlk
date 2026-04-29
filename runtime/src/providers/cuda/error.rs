use cudarc::driver::DriverError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("internal CUDA error: {0}")]
    Internal(#[from] DriverError),

    #[error("external CUDA error: {0}")]
    External(#[from] rmlk_cuda::Error),
}
