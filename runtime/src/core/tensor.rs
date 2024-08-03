use crate::core::utils;
use rmlk_ir::DataType;

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<T> {
    data: Option<T>,
    dtype: DataType,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(dtype: DataType) -> Self {
        Self {
            data: None,
            dtype,
            shape: Vec::new(),
            stride: Vec::new(),
        }
    }

    pub fn new_with_shape(dtype: DataType, shape: Vec<usize>) -> Self {
        let dims = shape.len();
        let mut stride = vec![0usize; dims];
        utils::calculate_stride(&shape, &mut stride);

        Self {
            data: None,
            dtype,
            shape,
            stride,
        }
    }

    pub fn _new_init(data: T, dtype: DataType, shape: Vec<usize>, stride: Vec<usize>) -> Self {
        Self {
            data: Some(data),
            dtype,
            shape,
            stride,
        }
    }

    pub fn init(&mut self, data: T) {
        self.data = Some(data);
    }

    pub fn _is_init(&self) -> bool {
        self.data.is_some()
    }

    pub fn data(&self) -> Option<&T> {
        self.data.as_ref()
    }

    pub fn data_mut(&mut self) -> Option<&mut T> {
        self.data.as_mut()
    }

    pub fn _take_data(&mut self) -> Option<T> {
        self.data.take()
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn _reshape(&mut self, shape: Vec<usize>) {
        self.shape = shape;
        let dims = self.shape.len();
        let mut stride = vec![0usize; dims];
        utils::calculate_stride(self.shape.as_slice(), &mut stride.as_mut_slice());
    }

    pub fn stride(&self) -> &Vec<usize> {
        &self.stride
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn set_dtype(&mut self, dtype: DataType) {
        self.dtype = dtype;
    }
}
