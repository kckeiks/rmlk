use crate::core::allocators::ShapeBufArena;
use crate::core::error::InternalError;
use crate::core::error::Result;
use crate::core::{device_service::DeviceService, Tensor};
use rmlk_graph::Graph;
use rmlk_schema::{DataType, Definition, Op};
use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::rc::Rc;

/// Tensor store.
///
/// This object stores a fixed-size collection of tensors.
pub struct TensorStore<T> {
    tensors: Box<[Option<RefCell<Tensor<T>>>]>,
}

impl<T> TensorStore<T> {
    pub fn new<D: DeviceService<Data = T>>(
        provider: &D,
        graph: &Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Result<TensorStore<D::Data>> {
        let shape_buf_arena = Rc::new(RefCell::new(ShapeBufArena::with_capacity(4096)));
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
            let index = shape_buf_arena
                .borrow_mut()
                .alloc_from_shape_slice(ir_tensor.dims.as_slice())?;

            let mut tensor =
                Tensor::new_with_shape(ir_tensor.data_type, index, shape_buf_arena.clone());
            tensor.set_dev_data(data);
            tensors[node_id].replace(RefCell::new(tensor));
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
                    let index = shape_buf_arena
                        .borrow_mut()
                        .alloc_from_shape_slice(shape.as_slice())?;
                    let tensor = Tensor::new_with_shape(dtype, index, shape_buf_arena.clone());
                    tensors[node_id].replace(RefCell::new(tensor));
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
                    let index = shape_buf_arena
                        .borrow_mut()
                        .alloc_from_shape_slice(shape.as_slice())?;
                    let tensor = Tensor::new_with_shape(dtype, index, shape_buf_arena.clone());
                    tensors[node_id].replace(RefCell::new(tensor));
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
                            tensors[*output].replace(RefCell::new(Tensor::new(DataType::Undefined, shape_buf_arena.clone())));
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
        })
    }

    pub fn get(&self, id: usize) -> Option<Ref<'_, Tensor<T>>> {
        self.tensors
            .get(id)
            .and_then(|x| x.as_ref().map(|y| y.borrow()))
    }

    pub fn get_mut(&self, id: usize) -> Option<RefMut<'_, Tensor<T>>> {
        self.tensors
            .get(id)
            .and_then(|x| x.as_ref().map(|y| y.borrow_mut()))
    }

    pub fn get_inner_mut(&mut self, id: usize) -> Option<&mut Tensor<T>> {
        self.tensors
            .get_mut(id)
            .and_then(|x| x.as_mut().map(|y| y.get_mut()))
    }
}

// Todo: Move to utils after refactor.
fn to_float_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
