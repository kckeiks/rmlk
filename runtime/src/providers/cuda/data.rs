use crate::core::device_service::DeviceData;
use anyhow::Result;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use rmlk_schema::{DataType, DataTypeMap};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[derive(Debug)]
pub struct CudaData {
    stream: Arc<CudaStream>,
    ptr: CUdeviceptr,
    len: usize,
    dtype: DataType,
    forget: bool,
}

impl CudaData {
    pub fn new<T>(dev_data: CudaSlice<T>) -> Self
    where
        T: DataTypeMap,
    {
        // Todo: For now we assume everything is in the default stream.
        let device = dev_data.context().default_stream();
        let len = dev_data.len();
        Self {
            stream: device,
            ptr: dev_data.leak(),
            len,
            dtype: T::data_type(),
            forget: false,
        }
    }

    pub fn data<T>(&self) -> DataView<T>
    where
        T: DataTypeMap,
    {
        if !self.is_dtype(T::data_type()) {
            panic!(
                "Cuda slice data type mismatch: received `{:?}` but expected `{:?}`",
                T::data_type(),
                self.dtype
            );
        }
        let slice = unsafe { self.stream.upgrade_device_ptr::<T>(self.ptr, self.len) };

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
            panic!(
                "Cuda slice data type mismatch: received `{:?}` but expected `{:?}`",
                T::data_type(),
                self.dtype
            );
        }
        let slice = unsafe { self.stream.upgrade_device_ptr::<T>(self.ptr, self.len) };

        DataViewMut {
            slice: Some(slice),
            _marker: PhantomData,
        }
    }

    #[inline]
    fn is_dtype(&self, other: DataType) -> bool {
        self.dtype == other
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn zero<T>(&mut self) -> Result<()>
    where
        T: DataTypeMap + ValidAsZeroBits + DeviceRepr,
    {
        let stream = self.stream.clone();
        stream.memset_zeros(self.data_mut::<T>().as_mut())?;
        Ok(())
    }

    // Note: The underlying CudaSlice will not be deallocated.
    pub fn forget(&mut self) {
        self.forget = true;
    }
}

impl Drop for CudaData {
    fn drop(&mut self) {
        if self.forget {
            return;
        }

        unsafe {
            match self.dtype {
                DataType::Float => {
                    let _dev_data = self.stream.upgrade_device_ptr::<f32>(self.ptr, self.len);
                }
                DataType::Int32 => {
                    let _dev_data = self.stream.upgrade_device_ptr::<i32>(self.ptr, self.len);
                }
                DataType::Int64 => {
                    let _dev_data = self.stream.upgrade_device_ptr::<i64>(self.ptr, self.len);
                }
                DataType::Bool => {
                    let _dev_data = self.stream.upgrade_device_ptr::<bool>(self.ptr, self.len);
                }
                DataType::USize => {
                    let _dev_data = self.stream.upgrade_device_ptr::<usize>(self.ptr, self.len);
                }
                dtype => unimplemented!("CudaDevData::drop unimplemented for `{dtype:?}`!"),
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

impl DeviceData for CudaData {}
