use crate::core::error::{InternalError, Result};
use crate::utils;
use bumpalo::Bump;
use log::warn;

pub struct ScratchAllocator {
    inner: Bump,
}

impl ScratchAllocator {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(size: usize) -> Self {
        Self {
            inner: Bump::with_capacity(size),
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
    {
        let target = self.inner.alloc_slice_fill_default::<T>(src.len());
        for (i, elem) in src.iter().enumerate() {
            target[i] = T::try_from(*elem).map_err(|_| InternalError::UnableToConvertValue)?;
        }
        Ok(target)
    }

    pub fn reset(&mut self) {
        self.inner.reset()
    }
}

pub struct ShapeBufArena {
    arena: Vec<usize>,
    current: usize,
    _max_size: usize,
}

impl ShapeBufArena {
    pub fn with_capacity(size: usize) -> Self {
        let arena = vec![0; size];
        Self {
            arena,
            current: 0,
            _max_size: size,
        }
    }

    pub fn alloc(&mut self, size: usize) -> Result<Index> {
        if self.arena.len() < self.current + 2 * size {
            warn!("Shape allocator was too small. Allocating more memory,");
            self.arena.resize(size, 0);
        }
        let old_current = self.current;
        self.current = old_current + size + size;
        Ok(Index {
            start: old_current,
            mid: old_current + size,
            end: old_current + size + size,
        })
    }

    pub fn alloc_from_shape_slice(&mut self, shape_src: &[usize]) -> Result<Index> {
        let size = shape_src.len();
        if self.arena.len() < self.current + 2 * size {
            warn!("Shape allocator was too small. Allocating more memory,");
            self.arena.resize(size, 0);
        }

        let shape = &mut self.arena[self.current..self.current + size];
        shape.copy_from_slice(shape_src);

        let stride = &mut self.arena[self.current + size..self.current + size + size];
        utils::calculate_stride(shape_src, stride);

        let old_current = self.current;
        self.current = old_current + size + size;
        Ok(Index {
            start: old_current,
            mid: old_current + size,
            end: old_current + size + size,
        })
    }

    pub fn copy_from_within(&mut self, src: &Index, dst: &Index) -> Result<()> {
        self.arena.copy_within(src.start..src.end, dst.start);
        Ok(())
    }

    pub fn alloc_and_copy_from_within(&mut self, src: &Index) -> Result<Index> {
        let index = self.alloc(src.mid - src.start)?;
        self.copy_from_within(src, &index)?;
        Ok(index)
    }

    pub fn get_shape_buf(&self, index: &Index) -> Option<&[usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&self.arena[index.start..index.mid])
    }

    pub fn get_stride_buf(&self, index: &Index) -> Option<&[usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&self.arena[index.mid..index.end])
    }

    pub fn get_shape_buf_mut(&mut self, index: &Index) -> Option<&mut [usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&mut self.arena[index.start..index.mid])
    }

    pub fn get_stride_buf_mut(&mut self, index: &Index) -> Option<&mut [usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&mut self.arena[index.mid..index.end])
    }
}

#[derive(Clone, Copy)]
pub struct Index {
    start: usize,
    mid: usize,
    end: usize,
}
