use crate::providers::cuda::allocator::CudaBump;
use crate::providers::cuda::data::{CudaData, DataView};
use crate::utils::Shape;
use anyhow::{anyhow, Result};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use log::debug;
use rmlk_schema::{DataType, DataTypeMap};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

#[derive(Clone)]
pub struct Tensor {
    shape: Rc<Shape>,
    payload: Rc<RefCell<Option<CudaData>>>,
    allocator: Rc<CudaBump>,
}

impl Tensor {
    pub fn new(
        shape: Rc<Shape>,
        payload: Rc<RefCell<Option<CudaData>>>,
        allocator: Rc<CudaBump>,
    ) -> Self {
        Self {
            shape,
            payload,
            allocator,
        }
    }

    pub fn dtype(&self) -> DataType {
        self.payload
            .borrow()
            .as_ref()
            .map(|data| data.dtype())
            .unwrap_or(DataType::Undefined)
    }

    pub fn is_scalar(&self) -> bool {
        self.shape().is_empty() && self.payload.borrow().as_ref().map(|d| d.len()).unwrap_or(0) == 1
    }

    pub fn shape_handle(&self) -> &Shape {
        &self.shape
    }

    pub fn shape(&self) -> Ref<'_, [usize]> {
        self.shape.shape()
    }

    pub fn stride(&self) -> Ref<'_, [usize]> {
        self.shape.stride()
    }

    pub fn rank(&self) -> usize {
        self.shape.shape().len()
    }

    pub fn copy_shape(&self, src: &Shape) {
        self.shape.copy_shape(src);
    }

    pub fn copy_shape_from_slice(&self, src: &[usize]) {
        self.shape.copy_shape_from_slice(src);
    }

    pub fn is_empty(&self) -> bool {
        self.payload.borrow().is_none() || self.payload.borrow().as_ref().unwrap().len() == 0
    }

    pub fn len(&self) -> usize {
        self.payload.borrow().as_ref().map(|d| d.len()).unwrap_or(0)
    }

    pub fn payload(&self) -> Ref<'_, CudaData> {
        Ref::filter_map(self.payload.borrow(), |p| p.as_ref())
            .expect("Caller to make sure payload exists")
    }

    pub fn payload_mut(&self) -> RefMut<'_, CudaData> {
        RefMut::filter_map(self.payload.borrow_mut(), |p| p.as_mut())
            .expect("Caller to make sure payload exists")
    }

    pub fn payload_to_vec<T>(&self) -> Result<Vec<T>>
    where
        T: DataTypeMap + DeviceRepr + Default + Clone,
    {
        let len = self.len();
        let mut buf = vec![T::default(); len];
        self.payload_to_host(&mut buf)?;
        Ok(buf)
    }

    pub fn payload_to_host<T>(&self, dst: &mut [T]) -> Result<()>
    where
        T: DataTypeMap + DeviceRepr,
    {
        assert_eq!(self.len(), dst.len());
        assert_eq!(self.dtype(), T::data_type());

        let ref_mut = self.payload.borrow();
        let cuda_data = ref_mut
            .as_ref()
            .ok_or_else(|| anyhow!("missing device data"))?;
        let data = cuda_data.data::<T>();
        let stream = data.stream().clone();
        Ok(stream.memcpy_dtoh(data.as_ref(), dst)?)
    }

    pub fn write_payload<T>(&self, src: &DataView<T>) -> Result<()>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        let mut ref_mut = self.payload.borrow_mut();

        let mut cuda_data = self.allocator.alloc_with_fallback::<T>(src.len())?;
        let stream = cuda_data.data::<T>().stream().clone();
        stream.memcpy_dtod(src.as_ref(), cuda_data.data_mut::<T>().as_mut())?;

        ref_mut.replace(cuda_data);

        Ok(())
    }

    pub fn write_payload_from_slice<T>(&self, src: &[T]) -> Result<()>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        let mut ref_mut = self.payload.borrow_mut();

        let mut cuda_data = self.allocator.alloc_with_fallback::<T>(src.len())?;
        let stream = cuda_data.data::<T>().stream().clone();
        stream.memcpy_htod(src, cuda_data.data_mut::<T>().as_mut())?;

        ref_mut.replace(cuda_data);

        Ok(())
    }

    fn init_payload_with_size<T>(&self, len: usize) -> Result<()>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        let mut ref_mut = self.payload.borrow_mut();
        let cuda_data = self.allocator.alloc_with_fallback::<T>(len)?;
        ref_mut.replace(cuda_data);
        Ok(())
    }

    pub fn init_scalar_payload<T>(&self) -> Result<()>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        self.init_payload_with_size::<T>(1)
    }

    pub fn init_payload<T>(&self) -> Result<()>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        let len = if !self.shape().is_empty() {
            self.shape().iter().product::<usize>()
        } else {
            panic!("zero-sized allocation are not allowed");
        };

        debug!("Allocating this much: {:?}", len);
        self.init_payload_with_size::<T>(len)
    }
}
