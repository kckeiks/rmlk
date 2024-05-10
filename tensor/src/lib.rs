#![feature(allocator_api)]
#![feature(ptr_as_uninit)]
#![feature(maybe_uninit_write_slice)]
#![feature(maybe_uninit_slice)]
#![allow(unused)]

// Todo: Remove allows.

mod alloc;
mod compute;
mod graph;
mod tensor;
mod tensorv2;
mod providers;

pub use tensor::op::Op;
