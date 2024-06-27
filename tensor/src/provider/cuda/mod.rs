pub mod cuda;
pub mod data;
pub mod kernels;
mod ops;

pub type Result<T> = std::result::Result<T, ()>;
