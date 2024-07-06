extern crate core;

mod cuda;
mod error;
mod execution_state;
mod kernel;
mod provider;
mod tensor;

pub use error::Error;
pub use execution_state::ExecutionState;
pub use kernel::{Context, Kernel};
pub use provider::Provider;
pub use tensor::Tensor;
type Result<T> = std::result::Result<T, Error>;
