mod attributes;
mod core;
mod providers;
mod utils;

#[cfg(test)]
mod testing;

pub use core::{Builder, ModelInstance, Value};

// Todo: define a MAX_RANK of 8.
