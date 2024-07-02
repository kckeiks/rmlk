use crate::provider::Provider;
use rmlk_ir::DataType;

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<P: Provider> {
    data: Option<P::Data>,
    dtype: DataType,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl<P> Tensor<P>
where
    P: Provider,
{
    pub fn new(dtype: DataType, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        Self {
            data: None,
            dtype,
            shape,
            stride,
        }
    }

    pub fn new_init(data: P::Data, dtype: DataType, shape: Vec<usize>, stride: Vec<usize>) -> Self {
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

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn reshape(&mut self, _: Vec<usize>) -> Result<(), ()> {
        todo!()
    }

    pub fn stride(&self) -> &Vec<usize> {
        &self.stride
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
}
