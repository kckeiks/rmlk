mod context;
mod cuda;
mod error;
mod kernel;
mod provider;
mod tensor;

pub use error::Error;
pub use provider::Provider;

type Result<T> = std::result::Result<T, Error>;
