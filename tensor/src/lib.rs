mod cuda;
mod error;
mod execution_state;
mod kernel;
mod provider;
mod tensor;

pub use error::Error;
pub use provider::Provider;

type Result<T> = std::result::Result<T, Error>;
