use crate::core::error::Result;
use crate::utils;
use rmlk_schema::DataType;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<T> {
    data: Option<Rc<RefCell<T>>>,
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

    pub fn set_dev_data(&mut self, data: T) -> Option<DevDataView<T>> {
        self.data
            .replace(Rc::new(RefCell::new(data)))
            .map(DevDataView)
    }

    pub fn dev_data(&self) -> Option<Ref<'_, T>> {
        self.data.as_ref().map(|data| data.borrow())
    }

    pub fn dev_data_mut(&self) -> Option<RefMut<'_, T>> {
        self.data.as_ref().map(|data| data.borrow_mut())
    }

    pub fn dev_data_view(&self) -> Option<DevDataView<T>> {
        self.data.as_ref().map(Clone::clone).map(DevDataView)
    }

    pub fn set_dev_data_from_view(&mut self, view: DevDataView<T>) -> Option<DevDataView<T>> {
        self.data.replace(view.0.clone()).map(DevDataView)
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

    pub fn reshape(&mut self, src: &[usize]) -> Result<()> {
        if src.len() != self.shape.len() {
            // Todo: Remove this once we pre-allocate these buffers.
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

pub struct DevDataView<T>(Rc<RefCell<T>>);
