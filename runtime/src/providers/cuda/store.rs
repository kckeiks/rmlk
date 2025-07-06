use crate::core::allocators::BufferArena;
use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::providers::cuda::data::{CudaData, DataView};
use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DeviceRepr};
use num_traits::ToPrimitive;
use rmlk_schema::DataTypeMap;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;

pub struct CudaBump {
    stream: Arc<CudaStream>,
    ptr: sys::CUdeviceptr,
    len: usize,
    cur_len: Cell<u64>,
}

impl CudaBump {
    pub fn new(stream: Arc<CudaStream>, size: usize) -> Result<Self> {
        let slice = stream
            .alloc_zeros::<u8>(size)
            .map_err(|e| InternalError::Device { error: e.into() })?;
        let len = slice.len();
        let ptr = slice.leak();
        Ok(Self {
            stream,
            ptr,
            len,
            cur_len: Cell::new(0),
        })
    }

    fn alloc<T: DataTypeMap>(&self, len: usize) -> Option<CudaData> {
        let align = align_of::<T>() as u64;
        let size = size_of::<T>() as u64;
        let mut cur = self.cur_len.get();

        cur = (cur + align - 1) & !(align - 1);

        let bytes_needed = (len as u64).checked_mul(size)?;
        let end = cur.checked_add(bytes_needed)?;

        if end > self.len as u64 {
            return None;
        }

        let slice = unsafe {
            self.stream
                .upgrade_device_ptr::<T>(self.ptr.checked_add(cur)?, len)
        };

        self.cur_len.set(end);

        let mut cuda_data = CudaData::new(slice);

        cuda_data.forget();

        Some(cuda_data)
    }
}

impl Drop for CudaBump {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stream.upgrade_device_ptr::<u8>(self.ptr, self.len);
        };
    }
}

#[derive(Clone)]
pub struct Slab {
    range: std::ops::Range<usize>,
    arena: Rc<RefCell<Vec<usize>>>,
}

impl Slab {
    pub fn slice(&self) -> Ref<'_, [usize]> {
        Ref::map(self.arena.borrow(), |t| &t[self.range.clone()])
    }

    pub fn slice_mut(&self) -> RefMut<'_, [usize]> {
        RefMut::map(self.arena.borrow_mut(), |t| &mut t[self.range.clone()])
    }
}

pub struct Tensor {
    shape: Slab,
    stride: Slab,
    payload: Rc<RefCell<Option<CudaData>>>,
}

impl Tensor {
    pub fn write<T>(&mut self, src: &DataView<T>) -> Result<()>
    where
        T: DataTypeMap,
    {
        let mut ref_mut = self.payload.borrow_mut();
        let cuda_data = ref_mut.as_mut().unwrap();
        let stream = cuda_data.data::<T>().stream().clone();
        stream
            .memcpy_dtod(src.as_ref(), cuda_data.data_mut::<T>().as_mut())
            .unwrap();

        Ok(())
    }

    pub fn write_from_buf<T>(&mut self, src: &[T]) -> Result<()>
    where
        T: DataTypeMap + DeviceRepr,
    {
        let mut ref_mut = self.payload.borrow_mut();
        let cuda_data = ref_mut.as_mut().unwrap();
        let stream = cuda_data.data::<T>().stream().clone();
        stream
            .memcpy_htod(src, cuda_data.data_mut::<T>().as_mut())
            .unwrap();

        Ok(())
    }

    pub fn update_shape(&mut self, src: &Slab) {
        let mut shape = self.shape.slice_mut();
        shape.copy_within(src.range.clone(), 0);
    }

    pub fn update_shape_from_slice(&mut self, src: &[usize]) {
        let mut shape = self.shape.slice_mut();
        shape.copy_from_slice(src);
    }
}

pub struct TensorStore {
    stream: Arc<CudaStream>,
    allocator: Rc<CudaBump>,
    tensors: Box<[Option<Tensor>]>,
    memory_usage: usize,
}

pub struct MainStore {
    inner: Rc<TensorStore>,
}

fn foo<T>(s: &CudaSlice<T>) {
    println!("len: {}", s.len());
}

fn bar<T>(s: &mut CudaSlice<T>) {
    println!("len: {}", s.len());
}

fn testing() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let bump = CudaBump::new(stream, 8).unwrap();

    let mut slab = bump.alloc::<f32>(4).unwrap();

    foo::<f32>(&slab.data::<f32>());

    bar::<f32>(&mut slab.data_mut::<f32>());
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, DevicePtr};

    #[test]
    fn alignment_correct_lengths() {
        let dev = CudaContext::new(0).unwrap();
        let stream = dev.default_stream();
        let bump = CudaBump::new(stream.clone(), 1024).unwrap();

        let a = bump.alloc::<u8>(3).unwrap();
        assert_eq!(a.len(), 3);

        let b = bump.alloc::<f64>(6).unwrap();
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn alignment_after_mixed_alloc() {
        let dev = CudaContext::new(0).unwrap();
        let stream = dev.default_stream();
        let bump = CudaBump::new(stream.clone(), 1024).unwrap();

        let a = bump.alloc::<u8>(1).unwrap();
        let (ptr_a, _) = a.data::<u8>().as_ref().device_ptr(&stream);

        let b = bump.alloc::<f64>(1).unwrap();
        let (ptr_b, _) = b.data::<f64>().as_ref().device_ptr(&stream);

        assert_eq!(
            ptr_a % 8,
            0,
            "address produced by CUDA should be aligned to 256 bytes"
        );
        assert_eq!(
            ptr_b % 8,
            0,
            "second allocation should respect natural 8-byte alignment"
        );
        assert!(ptr_b > ptr_a, "second slice must be at a higher address");
    }

    #[test]
    fn slices_do_not_overlap() {
        let dev = CudaContext::new(0).unwrap();
        let stream = dev.default_stream();
        let bump = CudaBump::new(stream.clone(), 1024).unwrap();

        let s1 = bump.alloc::<u32>(10).unwrap();
        let s2 = bump.alloc::<u16>(20).unwrap();

        let (p1, _) = s1.data::<u32>().device_ptr(&stream);
        let (p2, _) = s2.data::<u16>().device_ptr(&stream);

        assert!(
            p2 >= p1 + 40,
            "second slice must start after first slice ends"
        );
    }

    #[test]
    fn out_of_space_returns_none() {
        let dev = CudaContext::new(0).unwrap();
        let stream = dev.default_stream();
        let bump = CudaBump::new(stream.clone(), 1024).unwrap();

        assert!(bump.alloc::<u8>(1024).is_some());
        assert!(bump.alloc::<u8>(1).is_none());
    }
}
