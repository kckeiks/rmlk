use crate::utils;
use log::warn;
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::rc::Rc;

#[derive(Clone)]
pub struct ShapeAllocator {
    arena: Rc<RefCell<Vec<usize>>>,
    cur_len: Rc<Cell<usize>>,
}

impl ShapeAllocator {
    pub fn new(size: usize) -> Self {
        Self {
            arena: Rc::new(RefCell::new(Vec::with_capacity(size))),
            cur_len: Rc::new(Cell::new(0)),
        }
    }

    pub fn alloc(&self, len: usize) -> Shape {
        let mut inner = self.arena.borrow_mut();

        let cur_len = self.cur_len.get();
        let bytes_needed = cur_len + (2 * len);
        if bytes_needed > inner.len() {
            warn!("Buffer arena was too small. Allocating more memory,");
            // Todo: How much more should we resize?
            inner.resize(bytes_needed, 0);
        }

        let shape = cur_len..(cur_len + len);
        let stride = (cur_len + len)..(cur_len + 2 * len);
        self.cur_len.set(bytes_needed);

        Shape {
            alloc: self.clone(),
            shape: Cell::new(shape.into()),
            stride: Cell::new(stride.into()),
        }
    }

    pub fn alloc_from_slice(&self, src: &[usize]) -> Shape {
        let new_slab = self.alloc(src.len());
        new_slab.copy_shape_from_slice(src);
        new_slab
    }

    pub fn empty(&self) -> Shape {
        self.alloc(0)
    }

    pub fn clear(&self) {
        self.cur_len.set(0);
    }

    fn borrow(&self) -> Ref<'_, Vec<usize>> {
        self.arena.borrow()
    }

    fn borrow_mut(&self) -> RefMut<'_, Vec<usize>> {
        self.arena.borrow_mut()
    }
}

pub struct Shape {
    shape: Cell<Range>,
    stride: Cell<Range>,
    alloc: ShapeAllocator,
}

impl Shape {
    pub fn allocator(&self) -> &ShapeAllocator {
        &self.alloc
    }

    pub fn shape(&self) -> Ref<'_, [usize]> {
        let range: std::ops::Range<usize> = self.shape.get().into();
        Ref::map(self.alloc.borrow(), |t| &t[range])
    }

    pub fn shape_mut(&self) -> RefMut<'_, [usize]> {
        let range: std::ops::Range<usize> = self.shape.get().into();
        RefMut::map(self.alloc.borrow_mut(), |t| &mut t[range])
    }

    pub fn stride(&self) -> Ref<'_, [usize]> {
        let range: std::ops::Range<usize> = self.stride.get().into();
        Ref::map(self.alloc.borrow(), |t| &t[range])
    }

    pub fn stride_mut(&self) -> RefMut<'_, [usize]> {
        let range: std::ops::Range<usize> = self.stride.get().into();
        RefMut::map(self.alloc.borrow_mut(), |t| &mut t[range])
    }

    pub fn len(&self) -> usize {
        self.shape.get().len()
    }

    pub fn copy_shape(&self, shape: &Shape) {
        let src_shape = shape.shape.get();
        let src_stride = shape.stride.get();

        if self.len() == shape.len() {
            let dst_shape = self.shape.get();
            let dst_stride = self.stride.get();

            assert_eq!(dst_shape.len(), dst_stride.len());

            self.write(src_shape.into(), dst_shape.start);
            self.write(src_stride.into(), dst_stride.start);
        } else {
            let new_shape = self.alloc.alloc(shape.len());
            let dst_shape = new_shape.shape.get();
            let dst_stride = new_shape.stride.get();

            self.write(src_shape.into(), dst_shape.start);
            self.write(src_stride.into(), dst_stride.start);

            self.replace(new_shape);
        }
    }

    pub fn copy_shape_from_slice(&self, src: &[usize]) {
        if self.len() == src.len() {
            let dst_shape: std::ops::Range<usize> = self.shape.get().into();
            let dst_stride: std::ops::Range<usize> = self.stride.get().into();

            assert_eq!(dst_shape.len(), dst_stride.len());

            let mut inner = self.alloc.borrow_mut();
            inner[dst_shape].copy_from_slice(src);
            utils::compute_stride(src, &mut inner[dst_stride]);
        } else {
            let new_shape = self.alloc.alloc(src.len());
            let dst_shape: std::ops::Range<usize> = new_shape.shape.get().into();
            let dst_stride: std::ops::Range<usize> = new_shape.stride.get().into();

            let mut inner = self.alloc.borrow_mut();
            inner[dst_shape].copy_from_slice(src);
            utils::compute_stride(src, &mut inner[dst_stride]);

            self.replace(new_shape);
        }
    }

    fn replace(&self, shape: Shape) {
        self.shape.set(shape.shape.get());
        self.stride.set(shape.stride.get());
    }

    fn write(&self, src: std::ops::Range<usize>, start: usize) {
        let mut inner = self.alloc.borrow_mut();
        inner.copy_within(src, start);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Range {
    start: usize,
    end: usize,
}

impl Range {
    fn len(&self) -> usize {
        std::ops::Range::<usize>::from(*self).len()
    }
}

impl From<std::ops::Range<usize>> for Range {
    fn from(r: std::ops::Range<usize>) -> Self {
        Range {
            start: r.start,
            end: r.end,
        }
    }
}

impl From<Range> for std::ops::Range<usize> {
    fn from(r: Range) -> Self {
        r.start..r.end
    }
}
