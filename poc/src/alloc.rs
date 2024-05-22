use std::alloc::Allocator;

pub trait TensorAllocator: Clone {
    type MetadataAlloc: Allocator + Clone;
    type ComputeAlloc: Allocator + Clone;
    fn metadata_alloc(&self) -> Self::MetadataAlloc;
    fn compute_alloc(&self) -> Self::ComputeAlloc;
}

impl<A> TensorAllocator for A
where
    A: Allocator + Clone,
{
    type MetadataAlloc = Self;
    type ComputeAlloc = Self;

    fn metadata_alloc(&self) -> Self::MetadataAlloc {
        self.clone()
    }

    fn compute_alloc(&self) -> Self::ComputeAlloc {
        self.clone()
    }
}
