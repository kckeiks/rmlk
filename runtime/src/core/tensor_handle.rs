use crate::core::allocators::ArenaId;
use rmlk_schema::DataType;
use std::cell::RefCell;
use std::rc::Rc;

pub struct TensorHandle<T> {
    _dtype: DataType,
    on_dev_data: Rc<RefCell<Option<T>>>,
    arena_id: Option<ArenaId>,
}

impl<T> TensorHandle<T> {
    pub fn new(
        dtype: DataType,
        shape_buf_index: Option<ArenaId>,
        on_dev_data: Rc<RefCell<Option<T>>>,
    ) -> Self {
        Self {
            _dtype: dtype,
            arena_id: shape_buf_index,
            on_dev_data,
        }
    }

    pub fn arena_id(&self) -> Option<&ArenaId> {
        self.arena_id.as_ref()
    }

    pub fn set_arena_id(&mut self, id: ArenaId) {
        self.arena_id = Some(id);
    }

    pub fn on_dev_data(&self) -> Rc<RefCell<Option<T>>> {
        self.on_dev_data.clone()
    }
}

pub struct SrcTensorId(usize);

impl From<usize> for SrcTensorId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<SrcTensorId> for usize {
    fn from(value: SrcTensorId) -> Self {
        value.0
    }
}

pub struct DstTensorId(usize);

impl From<usize> for DstTensorId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<DstTensorId> for usize {
    fn from(value: DstTensorId) -> Self {
        value.0
    }
}
