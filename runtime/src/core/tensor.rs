use crate::core::error::Result;
use crate::utils;
use rmlk_schema::DataType;
use std::cell::{Ref, RefCell};

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<T> {
    data: Option<T>,
    dtype: DataType,
    shape: RefCell<Box<[usize]>>,
    stride: RefCell<Box<[usize]>>,
}

impl<T> Tensor<T> {
    pub fn new(dtype: DataType) -> Self {
        Self {
            data: None,
            dtype,
            shape: RefCell::new(Box::new([])),
            stride: RefCell::new(Box::new([])),
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
            shape: RefCell::new(shape.into_boxed_slice()),
            stride: RefCell::new(stride.into_boxed_slice()),
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

    pub fn shape(&self) -> Ref<'_, [usize]> {
        Ref::map(self.shape.borrow(), |borrow| borrow.as_ref())
    }

    pub fn stride(&self) -> Ref<'_, [usize]> {
        Ref::map(self.stride.borrow(), |borrow| borrow.as_ref())
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn set_dtype(&mut self, dtype: DataType) {
        self.dtype = dtype;
    }

    pub fn _reshape(&mut self, shape: Box<[usize]>) {
        *self.shape.borrow_mut() = shape;

        let dims = self.shape.borrow().as_ref().len();
        let mut stride = vec![0usize; dims];
        utils::calculate_stride(self.shape.borrow().as_ref(), &mut stride.as_mut_slice());

        *self.stride.borrow_mut() = stride.into_boxed_slice();
    }

    pub fn reshape(&self, src: &[usize]) -> Result<()> {
        let mut shape = self.shape.borrow_mut();
        if src.len() != shape.len() {
            //  Todo: Remove this once we pre-allocate these buffers.
            *shape = vec![0; src.len()].into_boxed_slice();
            *self.stride.borrow_mut() = vec![0; src.len()].into_boxed_slice();
            // return Err(InternalError::BufferSizeMismatch {
            //     expected: src.len(),
            //     actual: shape.len(),
            // });
        }
        shape.as_mut().copy_from_slice(src);
        utils::calculate_stride(src, self.stride.borrow_mut().as_mut());

        Ok(())
    }
}
