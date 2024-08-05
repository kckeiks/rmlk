mod data;
mod kernel;
mod provider;

pub use kernel::*;
pub use provider::CudaProvider;

use data::CudaData;

pub type CudaExecutionState = crate::core::ExecutionState<CudaData>;
