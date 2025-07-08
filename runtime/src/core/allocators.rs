use anyhow::Result;
use bumpalo::Bump;
use std::rc::Rc;

#[derive(Clone)]
pub struct ScratchAllocator {
    inner: Rc<Bump>,
}

impl ScratchAllocator {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(size: usize) -> Self {
        Self {
            inner: Rc::new(Bump::with_capacity(size)),
        }
    }

    pub fn allocate<T: Default>(&self, len: usize) -> Result<&mut [T]> {
        Ok(self.inner.alloc_slice_fill_default::<T>(len))
    }

    pub fn allocate_fill<T: Copy>(&self, len: usize, value: T) -> Result<&mut [T]> {
        Ok(self.inner.alloc_slice_fill_copy(len, value))
    }

    pub fn allocate_from_slice<T: Copy>(&self, src: &[T]) -> Result<&mut [T]> {
        Ok(self.inner.alloc_slice_copy(src))
    }

    /// Allocate a scratch buffer and copy slice into the buffer while
    /// converting each element.
    ///
    /// Warning: This allocator uses [`Bump`] which does not call
    /// `Drop` on the objects that it allocates.
    pub fn allocate_and_convert_from_slice<S, T>(&self, src: &[S]) -> Result<&mut [T]>
    where
        S: Copy,
        T: Default + TryFrom<S>,
        <T as TryFrom<S>>::Error: std::error::Error + Send + Sync + 'static,
    {
        let target = self.inner.alloc_slice_fill_default::<T>(src.len());
        for (i, elem) in src.iter().enumerate() {
            target[i] = T::try_from(*elem)?;
        }
        Ok(target)
    }

    pub fn reset(&mut self) {
        Rc::get_mut(&mut self.inner).unwrap().reset()
    }
}
