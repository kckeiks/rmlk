use crate::core::allocators::{BufferArena, ShapeBufArena, ShapeBufArenaMut};
use crate::core::device_service::DeviceData;
use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::core::tensor_handle::TensorHandle;
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
    tensors: Box<[Option<TensorHandle<T>>]>,
    buf_arena: BufferArena,
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
        let mut buf_arena = BufferArena::with_capacity(4096);
        let mut shape_buf_arena = ShapeBufArenaMut::new(&mut buf_arena);
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

            let arena_id = shape_buf_arena.alloc_from_shape_slice(ir_tensor.dims.as_slice())?;
            let tensor = TensorHandle::new(
                ir_tensor.data_type,
                Some(arena_id),
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
                    let arena_id = shape_buf_arena.alloc_from_shape_slice(shape.as_slice())?;
                    let tensor =
                        TensorHandle::new(dtype, Some(arena_id), Rc::new(RefCell::new(None)));
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
                    let arena_id = shape_buf_arena.alloc_from_shape_slice(shape.as_slice())?;
                    let tensor =
                        TensorHandle::new(dtype, Some(arena_id), Rc::new(RefCell::new(None)));
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
                            tensors[*output].replace(TensorHandle::new(DataType::Undefined, None, Rc::new(RefCell::new(None))));
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
            buf_arena,
        })
    }

    pub fn get<'a>(&'a self, id: usize) -> Option<Tensor<'a, T>> {
        let tensor_handle = self.tensors.get(id)?.as_ref()?;
        let shape_buf_arena = ShapeBufArena::<'a>::new(&self.buf_arena);
        let (shape, stride) = match tensor_handle.arena_id().as_ref() {
            None => (None, None),
            Some(index) => {
                let shape = shape_buf_arena.get_shape_buf(&index);
                let stride = shape_buf_arena.get_stride_buf(&index);
                (shape, stride)
            }
        };
        Some(Tensor::new(id, shape, stride, tensor_handle.data()))
    }

    pub fn copy_shape_from_within(&mut self, src_id: usize, dst_id: usize) -> Result<()> {
        let src_arena_id = self
            .try_get_tensor(src_id)?
            .arena_id()
            .copied()
            .ok_or(InternalError::TensorNotFound { node_id: src_id })?;
        let dst_arena_id = self.try_get_tensor(dst_id)?.arena_id().copied();

        let mut shape_buf_arena = ShapeBufArenaMut::new(&mut self.buf_arena);
        if dst_arena_id
            .map(|id| id.size() == src_arena_id.size())
            .unwrap_or(false)
        {
            shape_buf_arena.copy_shape_from_within(&src_arena_id, &dst_arena_id.unwrap());
        } else {
            let new_arena_id = shape_buf_arena.alloc_and_copy_shape_from_within(&src_arena_id)?;
            let dst = self.try_get_tensor_mut(dst_id)?;
            dst.set_arena_id(new_arena_id);
        }

        Ok(())
    }

    pub fn copy_shape_from_slice(&mut self, shape: &[usize], node_id: usize) -> Result<()> {
        let dst = self.try_get_tensor(node_id)?;
        let arena_id = dst.arena_id().copied();

        let mut shape_buf_arena = ShapeBufArenaMut::new(&mut self.buf_arena);

        if arena_id
            .map(|id| id.shape_len() == shape.len())
            .unwrap_or(false)
        {
            shape_buf_arena.try_copy_shape_from_slice(shape, &arena_id.unwrap())?;
        } else {
            let new_arena_id = shape_buf_arena.alloc_from_shape_slice(shape)?;
            let dst = self.try_get_tensor_mut(node_id)?;
            dst.set_arena_id(new_arena_id);
        }

        Ok(())
    }

    fn try_get_tensor(&self, id: usize) -> Result<&TensorHandle<T>> {
        self.tensors
            .get(id)
            .and_then(|handle| handle.as_ref())
            .ok_or(InternalError::TensorNotFound { node_id: id })
    }

    fn try_get_tensor_mut(&mut self, id: usize) -> Result<&mut TensorHandle<T>> {
        self.tensors
            .get_mut(id)
            .and_then(|handle| handle.as_mut())
            .ok_or(InternalError::TensorNotFound { node_id: id })
    }
}

// Todo: Move to utils after refactor.
fn to_float_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
