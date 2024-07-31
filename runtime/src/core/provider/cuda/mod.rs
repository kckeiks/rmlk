mod data;
mod kernel;
mod provider;

pub use kernel::*;
pub use provider::CudaProvider;

use crate::core::execution_state;
use data::CudaData;

pub type CudaExecutionState = execution_state::ExecutionState<CudaData>;
