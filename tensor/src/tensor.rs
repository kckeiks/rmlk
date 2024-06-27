use crate::provider::Provider;
use rmlk_hir::DataType;
use std::alloc::Allocator;

type Shape<A> = Vec<usize, A>;
type Stride<A> = Vec<usize, A>;

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<P: Provider> {
    data: Option<P::Data>,
    dtype: DataType,
    shape: Shape<P::Allocator>,
    stride: Stride<P::Allocator>,
}

impl<P> Tensor<P>
where
    P: Provider,
{
    pub fn new(dtype: DataType, shape: Shape<P::Allocator>, stride: Stride<P::Allocator>) -> Self {
        Self {
            data: None,
            dtype,
            shape,
            stride,
        }
    }

    pub fn new_init(
        data: P::Data,
        dtype: DataType,
        shape: Shape<P::Allocator>,
        stride: Stride<P::Allocator>,
    ) -> Self {
        Self {
            data: Some(data),
            dtype,
            shape,
            stride,
        }
    }

    pub fn init(&mut self, data: P::Data) {
        self.data = Some(data);
    }

    fn is_init(&self) -> bool {
        self.data.is_some()
    }

    pub fn data(&self) -> Option<&P::Data> {
        self.data.as_ref()
    }

    pub fn data_mut(&mut self) -> Option<&mut P::Data> {
        self.data.as_mut()
    }

    pub fn shape(&self) -> &Shape<P::Allocator> {
        &self.shape
    }

    pub fn reshape(&mut self, _: Shape<P::Allocator>) -> Result<(), ()> {
        todo!()
    }

    pub fn stride(&self) -> &Stride<P::Allocator> {
        &self.stride
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
}
