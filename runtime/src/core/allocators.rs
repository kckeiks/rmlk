use crate::core::error::{InternalError, Result};
use crate::utils;
use bumpalo::Bump;
use log::warn;
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
    {
        let target = self.inner.alloc_slice_fill_default::<T>(src.len());
        for (i, elem) in src.iter().enumerate() {
            target[i] = T::try_from(*elem).map_err(|_| InternalError::UnableToConvertValue)?;
        }
        Ok(target)
    }

    pub fn reset(&mut self) {
        Rc::get_mut(&mut self.inner).unwrap().reset()
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

    /// Allocates new buffers for shape and stride.
    /// Return the ID for the new buffers.
    pub fn alloc(&mut self, size: usize) -> Result<SbaId> {
        if self.arena.len() < self.current + 2 * size {
            warn!("Shape allocator was too small. Allocating more memory,");
            self.arena.resize(size, 0);
        }
        let old_current = self.current;
        self.current = old_current + size + size;
        Ok(SbaId {
            start: old_current,
            mid: old_current + size,
            end: old_current + size + size,
        })
    }

    /// Allocates new buffers and copies argument into the newly created buffers.
    /// Return the ID for the new buffers.
    pub fn alloc_from_shape_slice(&mut self, shape_src: &[usize]) -> Result<SbaId> {
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

        Ok(SbaId {
            start: old_current,
            mid: old_current + size,
            end: old_current + size + size,
        })
    }

    /// Allocates new buffers and copies argument into the newly created buffers.
    /// Returns an error if the size of the `src` and `dst` don't match.
    pub fn try_copy_shape_from_slice(&mut self, src: &[usize], dst: &SbaId) -> Result<()> {
        if src.len() != dst.shape_len() {
            return Err(InternalError::BufferSizeMismatch {
                expected: src.len(),
                actual: dst.shape_len(),
            });
        }

        self.get_shape_buf_mut(dst)
            .ok_or(InternalError::UnknownShapeBuffer { index: *dst })?
            .copy_from_slice(src);
        let stride = self
            .get_stride_buf_mut(dst)
            .expect("Stride buffer exists if a shape buffer exists");
        utils::calculate_stride(src, stride);
        Ok(())
    }

    /// Copies `src`'s buffers into `dst`.
    ///
    /// Panics if arguments do not have the same size.
    pub fn copy_shape_from_within(&mut self, src: &SbaId, dst: &SbaId) {
        self.arena.copy_within(src.start..src.end, dst.start);
    }

    /// Allocates new buffers and copies data from buffers within the arena.
    /// Return the ID for the new buffers.
    pub fn alloc_and_copy_shape_from_within(&mut self, src: &SbaId) -> Result<SbaId> {
        let index = self.alloc(src.mid - src.start)?;
        self.copy_shape_from_within(src, &index);
        Ok(index)
    }

    pub fn get_shape_buf(&self, index: &SbaId) -> Option<&[usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&self.arena[index.start..index.mid])
    }

    pub fn get_stride_buf(&self, index: &SbaId) -> Option<&[usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&self.arena[index.mid..index.end])
    }

    fn get_shape_buf_mut(&mut self, index: &SbaId) -> Option<&mut [usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&mut self.arena[index.start..index.mid])
    }

    fn get_stride_buf_mut(&mut self, index: &SbaId) -> Option<&mut [usize]> {
        if index.end > self.arena.len() {
            return None;
        }
        Some(&mut self.arena[index.mid..index.end])
    }
}

/// Shape buffer arena ID.
#[derive(Clone, Copy, Debug)]
pub struct SbaId {
    start: usize,
    mid: usize,
    end: usize,
}

impl SbaId {
    pub fn shape_len(&self) -> usize {
        self.mid - self.start
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }
}
