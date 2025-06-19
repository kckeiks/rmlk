use crate::core::device_service::DeviceData;
use crate::core::tensor_handle::{DstTensorId, SrcTensorId};
use anyhow::{anyhow, bail, Result};
use rmlk_schema::DataType;
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

/// Tensor.
///
/// This is simply a wrapper that holds a pointer to memory
/// on a device and other information about the tensor like shape,
/// datatype and stride.
pub struct Tensor<'a, T> {
    data: Rc<RefCell<Option<T>>>,
    shape: Option<&'a [usize]>,
    stride: Option<&'a [usize]>,
    id: usize,
}

impl<'a, T> Tensor<'a, T>
where
    T: DeviceData,
{
    pub fn new(
        id: usize,
        shape: Option<&'a [usize]>,
        stride: Option<&'a [usize]>,
        data: Rc<RefCell<Option<T>>>,
    ) -> Self {
        Self {
            data,
            id,
            shape,
            stride,
        }
    }

    pub fn set_dev_data(&mut self, data: T) {
        self.data.borrow_mut().replace(data);
    }

    pub fn dev_data_ptr(&self) -> Option<Ref<'_, T>> {
        Some(Ref::map(self.data.as_ref().borrow(), |data| {
            data.as_ref().unwrap()
        }))
    }

    pub fn try_dev_data_ptr(&self) -> Result<Ref<'_, T>> {
        {
            if self.data.as_ref().borrow().as_ref().is_none() {
                bail!("missing device data");
            }
        }

        Ok(self.dev_data_ptr().expect(""))
    }

    pub fn dev_data_ptr_mut(&self) -> Option<RefMut<'_, T>> {
        if self.data.as_ref().borrow().as_ref().is_none() {
            return None;
        }

        Some(RefMut::map(self.data.as_ref().borrow_mut(), |data| {
            data.as_mut().unwrap()
        }))
    }

    pub fn try_dev_data_ptr_mut(&self) -> Result<RefMut<'_, T>> {
        {
            if self.data.as_ref().borrow().as_ref().is_none() {
                bail!("missing device data");
            }
        }

        Ok(self.dev_data_ptr_mut().expect(""))
    }

    pub fn dev_data_ptr_clone(&self) -> Option<DevDataPtr<T>> {
        Some(DevDataPtr(self.data.clone()))
    }

    pub fn set_dev_data_ptr(&mut self, view: DevDataPtr<T>) {
        let taken = view
            .0
            .borrow_mut()
            .take()
            .expect("Empty DevDataPtr is never created");
        self.data.borrow_mut().replace(taken);
    }

    pub fn dtype(&self) -> DataType {
        self.data
            .borrow()
            .as_ref()
            .map(|data| data.dtype())
            .unwrap_or(DataType::Undefined)
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref().unwrap()
    }

    pub fn stride(&self) -> &[usize] {
        self.stride.as_ref().unwrap()
    }

    pub fn try_shape(&self) -> Result<&[usize]> {
        self.shape.ok_or(anyhow!("missing shape"))
    }

    pub fn try_stride(&self) -> Result<&[usize]> {
        self.stride.ok_or(anyhow!("missing strides"))
    }

    pub fn src_id(&self) -> SrcTensorId {
        self.id.into()
    }

    pub fn dst_id(self) -> DstTensorId {
        self.id.into()
    }
}

pub struct DevDataPtr<T>(Rc<RefCell<Option<T>>>);
