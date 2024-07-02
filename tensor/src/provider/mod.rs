mod cpu;
pub mod cuda;
mod error;
mod kernel;
mod provider;

pub use error::Error;
pub use provider::Provider;
type Result<T> = std::result::Result<T, Error>;
