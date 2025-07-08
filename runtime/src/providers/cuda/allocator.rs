use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::providers::cuda::data::CudaData;
use cudarc::driver::{sys, CudaStream, DeviceRepr, ValidAsZeroBits};
use rmlk_schema::DataTypeMap;
use std::cell::Cell;
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

        debug_assert!(
            (ptr as usize) % 256 == 0,
            "CudaBump base pointer is not 256-byte aligned; got {:#x}",
            ptr
        );

        Ok(Self {
            stream,
            ptr,
            len,
            cur_len: Cell::new(0),
        })
    }

    // Note: this does not have a fallback.
    pub fn alloc<T>(&self, len: usize) -> Option<CudaData>
    where
        T: DataTypeMap + DeviceRepr,
    {
        let align = align_of::<T>() as u64;

        debug_assert!(
            align <= 256,
            "CudaBump currently only supports alignments up to 256 bytes (requested = {align})"
        );

        let size = size_of::<T>() as u64;
        let mut cur = self.cur_len.get();

        cur = (cur + align - 1) & !(align - 1);

        let bytes_needed = (len as u64)
            .checked_mul(size)
            .expect("`bytes needed` not to exceed 64-bit capacity");
        let end = cur
            .checked_add(bytes_needed)
            .expect("size of cuda arena not to exceed 64-bit capacity");

        if end > self.len as u64 {
            return None;
        }

        let slice = unsafe {
            self.stream.upgrade_device_ptr::<T>(
                self.ptr
                    .checked_add(cur)
                    .expect("pointer to fit in a 64-bit address"),
                len,
            )
        };

        self.cur_len.set(end);

        let mut cuda_data = CudaData::new(slice);

        cuda_data.forget();

        Some(cuda_data)
    }

    pub fn alloc_from_slice_with_fallback<T>(&self, src: &[T]) -> Option<CudaData>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        let mut cuda_data = self.alloc_with_fallback::<T>(src.len())?;

        self.stream
            .memcpy_htod(src, cuda_data.data_mut().as_mut())
            .unwrap();

        Some(cuda_data)
    }

    pub fn alloc_with_fallback<T>(&self, len: usize) -> Option<CudaData>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        let cuda_data = match self.alloc::<T>(len) {
            Some(data) => data,
            None => {
                // Todo: we should return an error from this.
                self.alloc_fallback::<T>(len).unwrap()
            }
        };

        Some(cuda_data)
    }

    fn alloc_fallback<T>(&self, len: usize) -> Result<CudaData>
    where
        T: DataTypeMap + DeviceRepr + ValidAsZeroBits,
    {
        let raw_cuda_slice = self
            .stream
            .alloc_zeros::<T>(len)
            .map_err(|e| InternalError::Device { error: e.into() })?;
        Ok(CudaData::new(raw_cuda_slice))
    }

    pub fn clear(&self) {
        self.cur_len.set(0);
    }
}

impl Drop for CudaBump {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stream.upgrade_device_ptr::<u8>(self.ptr, self.len);
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::providers::cuda::allocator::CudaBump;
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
