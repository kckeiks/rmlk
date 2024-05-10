use std::alloc::Allocator;

pub trait Tensor {
    type Allocator: Allocator;

    fn init();

    fn allocator(&self) -> Self::Allocator;
}
