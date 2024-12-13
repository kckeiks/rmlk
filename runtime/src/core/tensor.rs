use crate::core::allocators::{Index, ShapeBufArena};
use crate::core::error::{InternalError, Result};
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
    shape_buf_index: Option<Index>,
    shape_buf_arena: Rc<RefCell<ShapeBufArena>>,
}

impl<T> Tensor<T> {
    pub fn new(dtype: DataType, shape_buf_arena: Rc<RefCell<ShapeBufArena>>) -> Self {
        Self {
            data: None,
            dtype,
            shape_buf_index: None,
            shape_buf_arena,
        }
    }

    pub fn new_with_shape(
        dtype: DataType,
        shape_buf_index: Index,
        shape_buf_arena: Rc<RefCell<ShapeBufArena>>,
    ) -> Self {
        Self {
            data: None,
            dtype,
            shape_buf_index: Some(shape_buf_index),
            shape_buf_arena,
        }
    }

    pub fn set_dev_data(&mut self, data: T) -> Option<DevDataPtr<T>> {
        self.data
            .replace(Rc::new(RefCell::new(data)))
            .map(DevDataPtr)
    }

    pub fn dev_data_ptr(&self) -> Option<Ref<'_, T>> {
        self.data.as_ref().map(|data| data.borrow())
    }

    pub fn try_dev_data_ptr(&self) -> Result<Ref<'_, T>> {
        self.data
            .as_ref()
            .map(|data| data.borrow())
            .ok_or(InternalError::MissingDeviceData)
    }

    pub fn dev_data_ptr_mut(&self) -> Option<RefMut<'_, T>> {
        self.data.as_ref().map(|data| data.borrow_mut())
    }

    pub fn try_dev_data_ptr_mut(&self) -> Result<RefMut<'_, T>> {
        self.data
            .as_ref()
            .map(|data| data.borrow_mut())
            .ok_or(InternalError::MissingDeviceData)
    }

    pub fn dev_data_ptr_clone(&self) -> Option<DevDataPtr<T>> {
        self.data.as_ref().map(Clone::clone).map(DevDataPtr)
    }

    pub fn set_dev_data_ptr(&mut self, view: DevDataPtr<T>) -> Option<DevDataPtr<T>> {
        self.data.replace(view.0.clone()).map(DevDataPtr)
    }

    pub fn shape(&self) -> Ref<[usize]> {
        let index = self.shape_buf_index.as_ref().expect("");
        Ref::map(self.shape_buf_arena.borrow(), |shape_buf| {
            shape_buf.get_shape_buf(index).expect("")
        })
    }

    pub fn stride(&self) -> Ref<[usize]> {
        let index = self.shape_buf_index.as_ref().unwrap();
        Ref::map(self.shape_buf_arena.borrow(), |shape_buf| {
            shape_buf.get_stride_buf(index).expect("")
        })
    }

    pub fn try_shape(&self) -> Result<Ref<[usize]>> {
        let index = self
            .shape_buf_index
            .as_ref()
            .ok_or(InternalError::MissingDeviceData)?;
        Ok(Ref::map(self.shape_buf_arena.borrow(), |shape_buf| {
            shape_buf.get_shape_buf(index).expect("")
        }))
    }

    pub fn try_stride(&self) -> Result<Ref<[usize]>> {
        let index = self
            .shape_buf_index
            .as_ref()
            .ok_or(InternalError::MissingDeviceData)?;
        Ok(Ref::map(self.shape_buf_arena.borrow(), |shape_buf| {
            shape_buf.get_stride_buf(index).expect("")
        }))
    }

    fn try_shape_mut(&mut self) -> Result<RefMut<[usize]>> {
        let index = self
            .shape_buf_index
            .as_ref()
            .ok_or(InternalError::MissingDeviceData)?;
        Ok(RefMut::map(
            self.shape_buf_arena.borrow_mut(),
            |shape_buf| shape_buf.get_shape_buf_mut(index).expect(""),
        ))
    }

    fn try_stride_mut(&mut self) -> Result<RefMut<[usize]>> {
        let index = self
            .shape_buf_index
            .as_ref()
            .ok_or(InternalError::MissingDeviceData)?;
        Ok(RefMut::map(
            self.shape_buf_arena.borrow_mut(),
            |shape_buf| shape_buf.get_stride_buf_mut(index).expect(""),
        ))
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn set_dtype(&mut self, dtype: DataType) {
        self.dtype = dtype;
    }

    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<()> {
        let good = {
            self.try_shape().map(|shape| shape.len() != new_shape.len()).unwrap_or(true)
        };
        if good{
            let mut arena = self.shape_buf_arena.borrow_mut();
            self.shape_buf_index = Some(arena.alloc_from_shape_slice(new_shape)?);
        }
        Ok(())
    }

    pub fn copy_shape(&mut self, new_shape: Index) -> Result<()> {
        let mut arena = self.shape_buf_arena.borrow_mut();
        self.shape_buf_index = Some(arena.alloc_and_copy_from_within(&new_shape)?);
        Ok(())
    }

    pub fn try_index(&self) -> Result<Index> {
        self.shape_buf_index.ok_or(InternalError::MissingDeviceData)
    }
}

pub struct DevDataPtr<T>(Rc<RefCell<T>>);
