use crate::core::allocators::{Index, ShapeBufArena};
use crate::core::device_service::DeviceData;
use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::core::{device_service::DeviceService, Tensor};
use rmlk_graph::Graph;
use rmlk_schema::{DataType, Definition, Op};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Tensor store.
///
/// This object stores a fixed-size collection of tensors.
pub struct TensorStore<T> {
    tensors: Box<[Option<TensorBacking<T>>]>,
    shape_buf_arena: ShapeBufArena,
}

impl<T> TensorStore<T>
where
    T: DeviceData,
{
    pub fn new<D: DeviceService<Data = T>>(
        provider: &D,
        graph: &Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Result<TensorStore<D::Data>> {
        let mut shape_buf_arena = ShapeBufArena::with_capacity(4096);
        // Todo: We might need the max id of the graph instead.
        let node_count = graph.node_count();
        let mut tensors = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            tensors.push(None);
        }

        // Todo: When we fix the traversal procedure, we will have the set of nodes that
        // need values and we can avoid these overlapping loops.

        // Load initializers.
        for (node_id, ir_tensor) in initializers {
            debug_assert_eq!(
                graph.get_node(node_id).map(|n| n.value().op()),
                Some(Op::Const)
            );

            let data = match ir_tensor.float_data.is_empty() {
                true => to_float_vec(ir_tensor.raw_data.as_ref().ok_or_else(|| {
                    InternalError::TensorStore("failed to parse tensor raw data".to_string())
                })?),
                false => {
                    // Todo: remove allocation.
                    ir_tensor.float_data.clone()
                }
            };
            let data = provider.htod_float(data)?;
            let index = StoreIndex::with_arena_index(
                node_id,
                shape_buf_arena.alloc_from_shape_slice(ir_tensor.dims.as_slice())?,
            );

            let tensor = TensorBacking::new(
                ir_tensor.data_type,
                node_id,
                index,
                Rc::new(RefCell::new(Some(data))),
            );
            tensors[node_id].replace(tensor);
        }

        for node_id in graph.inputs() {
            match graph.get_node(node_id) {
                Some(node) => {
                    let def = node.value();
                    let dtype = def
                        .dtype()
                        .ok_or(InternalError::ExpectedDataTypeInDef { node_id })?;
                    let shape = def
                        .shape()
                        .ok_or(InternalError::ExpectedShapeInDef { node_id })?;
                    let index = StoreIndex::with_arena_index(
                        node_id,
                        shape_buf_arena.alloc_from_shape_slice(shape.as_slice())?,
                    );
                    let tensor =
                        TensorBacking::new(dtype, node_id, index, Rc::new(RefCell::new(None)));
                    tensors[node_id].replace(tensor);
                }
                None => {
                    return Err(InternalError::TensorStore(format!(
                        "failed to create tensor store: failed to find input node `{node_id}`"
                    )))
                }
            }
        }

        for node_id in graph.outputs() {
            match graph.get_node(node_id) {
                Some(node) => {
                    let def = node.value();
                    let dtype = def
                        .dtype()
                        .ok_or(InternalError::ExpectedDataTypeInDef { node_id })?;
                    let shape = def
                        .shape()
                        .ok_or(InternalError::ExpectedShapeInDef { node_id })?;
                    let index = StoreIndex::with_arena_index(
                        node_id,
                        shape_buf_arena.alloc_from_shape_slice(shape.as_slice())?,
                    );
                    let tensor =
                        TensorBacking::new(dtype, node_id, index, Rc::new(RefCell::new(None)));
                    tensors[node_id].replace(tensor);
                }
                None => {
                    return Err(InternalError::TensorStore(format!(
                        "failed to create tensor store: failed to find output node `{node_id}`"
                    )))
                }
            }
        }

        for (node_id, node) in graph.node_iter() {
            // We already loaded the initializers.
            if matches!(node.value().op(), Op::Const | Op::NoOp) {
                continue;
            }

            for output in node.outputs() {
                match graph.get_node(*output) {
                    Some(_) => {
                        if tensors.get(*output)
                            .ok_or_else(|| {
                                InternalError::TensorStore(
                                    format!("failed to create tensor store: node {} is referring to an output node ID that is unknown", *output)
                                )
                            })?
                            .is_none()
                        {
                            tensors[*output].replace(TensorBacking::new(DataType::Undefined, *output, StoreIndex::new(*output), Rc::new(RefCell::new(None))));
                        }
                    }
                    None => {
                        return Err(InternalError::TensorStore(format!(
                            "failed to create tensor store: failed to find output node `{}` for node {}",
                            *output,
                            node_id
                        )))
                    }
                }
            }
        }

        Ok(TensorStore {
            tensors: tensors.into_boxed_slice(),
            shape_buf_arena,
        })
    }

    pub fn get(&self, id: usize) -> Option<Tensor<T>> {
        let backing = self.tensors.get(id)?.as_ref()?;
        let (shape, stride) = match backing.shape_buf_index.arena_index.as_ref() {
            None => (None, None),
            Some(index) => {
                let shape = self.shape_buf_arena.get_shape_buf(&index);
                let stride = self.shape_buf_arena.get_stride_buf(&index);
                (shape, stride)
            }
        };
        Some(Tensor::new(
            backing.shape_buf_index,
            shape,
            stride,
            backing.data.clone(),
        ))
    }

    pub fn copy_within(&mut self, src: StoreIndex, dst: StoreIndex) -> Result<Tensor<T>> {
        let src = src
            .arena_index
            .as_ref()
            .ok_or(InternalError::MissingDeviceData)?;
        {
            let dst = self
                .tensors
                .get(dst.node_id)
                .ok_or(InternalError::MissingDeviceData)?
                .as_ref()
                .ok_or(InternalError::MissingDeviceData)?;
            match dst.shape_buf_index.arena_index.as_ref() {
                None => {
                    let index = self.shape_buf_arena.alloc_and_copy_from_within(src)?;
                    let dst = self
                        .tensors
                        .get_mut(dst.node_id)
                        .ok_or(InternalError::MissingDeviceData)?
                        .as_mut()
                        .ok_or(InternalError::MissingDeviceData)?;
                    dst.shape_buf_index.arena_index = Some(index);
                }
                Some(index) => {
                    self.shape_buf_arena.copy_from_within(src, index)?;
                }
            }
        }

        Ok(self.get(dst.node_id).unwrap())
    }

    pub fn copy_from_slice(&mut self, src: &[usize], dst: StoreIndex) -> Result<Tensor<T>> {
        {
            let dst = self
                .tensors
                .get_mut(dst.node_id)
                .ok_or(InternalError::MissingDeviceData)?
                .as_mut()
                .ok_or(InternalError::MissingDeviceData)?;
            match dst.shape_buf_index.arena_index.as_ref().copied() {
                None => {
                    // We need to allocate first.
                    let index = self.shape_buf_arena.alloc_from_shape_slice(src)?;
                    dst.shape_buf_index.arena_index = Some(index);
                }
                Some(index) => {
                    self.shape_buf_arena
                        .try_copy_shape_from_slice(src, &index)?;
                }
            }
        }

        Ok(self.get(dst.node_id).unwrap())
    }
}

// Todo: Move to utils after refactor.
fn to_float_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[derive(Clone, Copy)]
pub struct StoreIndex {
    node_id: usize,
    arena_index: Option<Index>,
}

impl StoreIndex {
    pub fn new(node_id: usize) -> Self {
        Self::inner_new(node_id, None)
    }

    pub fn with_arena_index(node_id: usize, arena_index: Index) -> Self {
        Self::inner_new(node_id, Some(arena_index))
    }

    fn inner_new(node_id: usize, arena_index: Option<Index>) -> Self {
        Self {
            node_id,
            arena_index,
        }
    }
}

pub struct TensorBacking<T> {
    _dtype: DataType,
    node_id: usize,
    data: Rc<RefCell<Option<T>>>,
    shape_buf_index: StoreIndex,
}

impl<T> TensorBacking<T> {
    pub fn new(
        dtype: DataType,
        node_id: usize,
        shape_buf_index: StoreIndex,
        data: Rc<RefCell<Option<T>>>,
    ) -> Self {
        Self {
            _dtype: dtype,
            node_id,
            data,
            shape_buf_index,
        }
    }
}
