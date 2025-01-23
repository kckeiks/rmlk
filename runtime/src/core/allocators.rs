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
    inner: BufferArena,
}

impl ShapeBufArena {
    pub fn with_capacity(size: usize) -> Self {
        Self {
            inner: BufferArena::with_capacity(size),
        }
    }

    /// Allocates new buffers for shape and stride.
    /// Return the ID for the new buffers.
    pub fn alloc(&mut self, size: usize) -> Result<ArenaId> {
        let key = self.inner.alloc(2 * size)?;
        Ok(ArenaId {
            key,
            size,
        })
    }

    /// Allocates new buffers and copies argument into the newly created buffers.
    /// Return the ID for the new buffers.
    pub fn alloc_from_shape_slice(&mut self, shape_src: &[usize]) -> Result<ArenaId> {
        let size = shape_src.len();

        let arena_id = self.alloc(size)?;

        let slab = self.inner.get_mut(arena_id.key).expect("We allocated before this call");

        // Copy the shape.
        slab[..arena_id.size].as_mut().copy_from_slice(shape_src);

        // Compute the stride.
        let stride =  slab[arena_id.size..].as_mut();
        utils::calculate_stride(shape_src, stride);

        Ok(arena_id)
    }

    /// Allocates new buffers and copies argument into the newly created buffers.
    /// Returns an error if the size of the `src` and `dst` don't match.
    pub fn try_copy_shape_from_slice(&mut self, src: &[usize], dst: &ArenaId) -> Result<()> {
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
    pub fn copy_shape_from_within(&mut self, src: &ArenaId, dst: &ArenaId) {
        self.inner.copy_within(src.key, dst.key);
    }

    /// Allocates new buffers and copies data from buffers within the arena.
    /// Return the ID for the new buffers.
    pub fn alloc_and_copy_shape_from_within(&mut self, src: &ArenaId) -> Result<ArenaId> {
        let index = self.alloc(src.size)?;
        self.copy_shape_from_within(src, &index);
        Ok(index)
    }

    pub fn get_shape_buf(&self, id: &ArenaId) -> Option<&[usize]> {
        let buf = self.inner.get(id.key)?;
        Some(&buf[..id.size])
    }

    pub fn get_stride_buf(&self, id: &ArenaId) -> Option<&[usize]> {
        let buf = self.inner.get(id.key)?;
        Some(&buf[id.size..])
    }

    fn get_shape_buf_mut(&mut self, id: &ArenaId) -> Option<&mut [usize]> {
        let buf = self.inner.get_mut(id.key)?;
        Some(&mut buf[..id.size])
    }

    fn get_stride_buf_mut(&mut self, id: &ArenaId) -> Option<&mut [usize]> {
        let buf = self.inner.get_mut(id.key)?;
        Some(&mut buf[id.size..])
    }
}

/// Shape-buffer arena ID.
#[derive(Clone, Copy, Debug)]
pub struct ArenaId {
    key: usize,
    size: usize,
}

impl ArenaId {
    pub fn shape_len(&self) -> usize {
        self.size
    }

    pub fn size(&self) -> usize {
        2 * self.size
    }
}

pub struct BufferArena {
    arena: Vec<usize>,
    current: usize,
}

impl BufferArena {
    const LEN_HEADER_SIZE: usize = 1;

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: vec![0; capacity],
            current: 0,
        }
    }

    pub fn alloc(&mut self, size: usize) -> Result<usize> {
        if size == 0 {
            return Err(InternalError::InvalidMemoryAllocation {
                message: "cannot allocate a buffer of size `0`".to_string(),
            });
        }

        let new_len = self.current + size + Self::LEN_HEADER_SIZE;

        if new_len > self.arena.len() {
            warn!("Buffer arena was too small. Allocating more memory,");
            // Todo: How much more should we resize?
            self.arena.resize(new_len, 0);
        }

        let old_current = self.current;
        self.current = new_len;
        self.arena[old_current] = size;

        Ok(old_current)
    }

    pub fn get(&self, key: usize) -> Option<&[usize]> {
        let size = self.arena.get(key).copied()?;

        let content_start = key + Self::LEN_HEADER_SIZE;

        assert_eq!(size, self.arena[content_start..content_start + size].len());

        Some(self.arena[content_start..content_start + size].as_ref())
    }

    pub fn get_mut(&mut self, key: usize) -> Option<&mut [usize]> {
        let size = self.arena.get(key).copied()?;

        let content_start = key + Self::LEN_HEADER_SIZE;

        assert_eq!(size, self.arena[content_start..content_start + size].len());

        Some(self.arena[content_start..content_start + size].as_mut())
    }

    pub fn copy_within(&mut self, src: usize, dst: usize) {
        let src_len = self.arena[src];
        let dst_len = self.arena[dst];

        assert_eq!(src_len, dst_len);

        let src_content_start = src + Self::LEN_HEADER_SIZE;
        let dst_content_start = dst + Self::LEN_HEADER_SIZE;

        self.arena.copy_within(src_content_start..src_content_start + src_len, dst_content_start);
    }
}