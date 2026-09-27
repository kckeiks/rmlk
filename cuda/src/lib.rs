// Low-level CUDA/cuDNN launch functions intentionally mirror native APIs.
#![allow(clippy::too_many_arguments)]
// Kernel launch safety contracts are enforced by their typed wrappers and call sites.
#![allow(clippy::missing_safety_doc)]

mod error;
pub mod kernels;
pub mod params;
mod ptx;
mod utils;

pub use error::Error;
