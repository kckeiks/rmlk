use crate::core::device_service::DeviceData;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use rmlk_schema::{DataType, DataTypeMap};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub struct CudaData {
    device: Arc<CudaDevice>,
    ptr: CUdeviceptr,
    len: usize,
    dtype: DataType,
}

impl CudaData {
    pub fn new<T>(dev_data: CudaSlice<T>) -> Self
    where
        T: DataTypeMap,
    {
        let device = dev_data.device();
        let len = dev_data.len();
        Self {
            device,
            ptr: dev_data.leak(),
            len,
            dtype: T::data_type(),
        }
    }

    pub fn data<T>(&self) -> DataView<T>
    where
        T: DataTypeMap,
    {
        if !self.is_dtype(T::data_type()) {
            panic!("Cuda slice data type mismatch");
        }
        let slice = unsafe { self.device.upgrade_device_ptr::<T>(self.ptr, self.len) };

        DataView {
            slice: Some(slice),
            _marker: PhantomData,
        }
    }

    pub fn data_mut<T>(&mut self) -> DataViewMut<T>
    where
        T: DataTypeMap,
    {
        if !self.is_dtype(T::data_type()) {
            panic!("Cuda slice data type mismatch");
        }
        let slice = unsafe { self.device.upgrade_device_ptr::<T>(self.ptr, self.len) };

        DataViewMut {
            slice: Some(slice),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn is_dtype(&self, other: DataType) -> bool {
        self.dtype == other
    }
}

impl Drop for CudaData {
    fn drop(&mut self) {
        unsafe {
            match self.dtype {
                DataType::Float => {
                    let _dev_data = self.device.upgrade_device_ptr::<f32>(self.ptr, self.len);
                }
                _ => unimplemented!("CudaDevData::drop unimplemented!"),
            }
        }
    }
}

pub struct DataView<'a, T> {
    slice: Option<CudaSlice<T>>,
    _marker: PhantomData<&'a CudaData>,
}

impl<T> AsRef<CudaSlice<T>> for DataView<'_, T> {
    fn as_ref(&self) -> &CudaSlice<T> {
        self.slice
            .as_ref()
            .expect("CudaSlice is null only after being dropped")
    }
}

impl<T> Deref for DataView<'_, T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &Self::Target {
        self.slice
            .as_ref()
            .expect("CudaSlice is null only after being dropped")
    }
}

impl<'a, T> Drop for DataView<'a, T> {
    fn drop(&mut self) {
        self.slice
            .take()
            .expect("CudaSlice is null only after being dropped")
            .leak();
    }
}

pub struct DataViewMut<'a, T> {
    slice: Option<CudaSlice<T>>,
    _marker: PhantomData<&'a mut CudaData>,
}

impl<T> AsRef<CudaSlice<T>> for DataViewMut<'_, T> {
    fn as_ref(&self) -> &CudaSlice<T> {
        self.slice
            .as_ref()
            .expect("CudaSlice is null only after being dropped")
    }
}

impl<'a, T> AsMut<CudaSlice<T>> for DataViewMut<'a, T> {
    fn as_mut(&mut self) -> &mut CudaSlice<T> {
        self.slice
            .as_mut()
            .expect("CudaSlice is null only after being dropped")
    }
}

impl<T> Deref for DataViewMut<'_, T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &Self::Target {
        self.slice
            .as_ref()
            .expect("CudaSlice is null only after being dropped")
    }
}

impl<T> DerefMut for DataViewMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
            .as_mut()
            .expect("CudaSlice is null only after being dropped")
    }
}

impl<'a, T> Drop for DataViewMut<'a, T> {
    fn drop(&mut self) {
        self.slice
            .take()
            .expect("CudaSlice is null only after being dropped")
            .leak();
    }
}

impl DeviceData for CudaData {
    fn dtype(&self) -> DataType {
        self.dtype
    }
}
