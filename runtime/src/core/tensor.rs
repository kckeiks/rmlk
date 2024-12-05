use crate::core::error::Result;
use crate::utils;
use rmlk_schema::DataType;

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<T> {
    data: Option<T>,
    dtype: DataType,
    shape: Box<[usize]>,
    stride: Box<[usize]>,
}

impl<T> Tensor<T> {
    pub fn new(dtype: DataType) -> Self {
        Self {
            data: None,
            dtype,
            shape: Box::new([]),
            stride: Box::new([]),
        }
    }

    // Todo: Update when we have a special allocator for long-lived data.
    pub fn new_with_shape(dtype: DataType, shape: Vec<usize>) -> Self {
        let dims = shape.len();
        let mut stride = vec![0usize; dims];
        utils::calculate_stride(&shape, &mut stride);

        Self {
            data: None,
            dtype,
            shape: shape.into_boxed_slice(),
            stride: stride.into_boxed_slice(),
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

    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }

    pub fn stride(&self) -> &[usize] {
        self.stride.as_ref()
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn set_dtype(&mut self, dtype: DataType) {
        self.dtype = dtype;
    }

    pub fn _reshape(&mut self, shape: Box<[usize]>) {
        self.shape = shape;

        let dims = self.shape.as_ref().len();
        let mut stride = vec![0usize; dims];
        utils::calculate_stride(self.shape.as_ref(), &mut stride.as_mut_slice());

        self.stride = stride.into_boxed_slice();
    }

    pub fn reshape(&mut self, src: &[usize]) -> Result<()> {
        if src.len() != self.shape.len() {
            //  Todo: Remove this once we pre-allocate these buffers.
            self.shape = vec![0; src.len()].into_boxed_slice();
            self.stride = vec![0; src.len()].into_boxed_slice();
            // return Err(InternalError::BufferSizeMismatch {
            //     expected: src.len(),
            //     actual: shape.len(),
            // });
        }
        self.shape.as_mut().copy_from_slice(src);
        utils::calculate_stride(src, self.stride.as_mut());

        Ok(())
    }
}
