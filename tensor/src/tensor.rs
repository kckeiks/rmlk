use crate::device::Device;
use crate::dtype::DType;
use std::sync::Arc;

type Shape<A> = Vec<usize, A>;
type Stride<A> = Vec<usize, A>;

pub struct Tensor<S: Device> {
    data: Option<Arc<S::Data>>,
    dtype: DType,
    shape: Shape<S::Cpu>,
    stride: Stride<S::Cpu>,
}

impl<S> Tensor<S>
where
    S: Device,
{
    pub fn new(dtype: DType, shape: Shape<S::Cpu>, stride: Stride<S::Cpu>) -> Self {
        Self {
            data: None,
            dtype,
            shape,
            stride,
        }
    }

    pub fn new_with_data(
        data: Arc<S::Data>,
        dtype: DType,
        shape: Shape<S::Cpu>,
        stride: Stride<S::Cpu>,
    ) -> Self {
        Self {
            data: Some(data),
            dtype,
            shape,
            stride,
        }
    }

    pub fn set_data(&mut self, data: Arc<S::Data>) -> Option<Arc<S::Data>> {
        self.data.replace(data)
    }

    pub fn data(&self) -> Option<&S::Data> {
        self.data.as_ref().map(|data| data.as_ref())
    }

    pub fn data_mut(&mut self) -> &mut S::Data {
        Arc::make_mut(self.data.as_mut().unwrap())
    }

    pub fn shape(&self) -> &Shape<S::Cpu> {
        &self.shape
    }

    pub fn stride(&self) -> &Stride<S::Cpu> {
        &self.stride
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}
