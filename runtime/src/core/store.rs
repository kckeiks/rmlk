use crate::core::allocators::{BufferArena, ShapeBufArena, ShapeBufArenaMut};
use crate::core::device_service::DeviceData;
use crate::core::tensor_handle::TensorHandle;
use crate::core::{device_service::DeviceService, Tensor};
use crate::utils::FromBytes;
use anyhow::Result;
use half::f16;
use log::debug;
use rmlk_graph::Graph;
use rmlk_schema::{DataType, Definition, Op};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
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
        Builder::new(provider, graph, initializers).build()
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
        Some(Tensor::new(id, shape, stride, tensor_handle.on_dev_data()))
    }

    pub fn copy_shape_from_within(&mut self, src_id: usize, dst_id: usize) -> Result<()> {
        let src_arena_id = self
            .try_get_tensor(src_id)?
            .arena_id()
            .copied()
            .ok_or(StoreError::TensorNotFound { id: src_id })?;
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
            .ok_or(StoreError::TensorNotFound { id })
            .map_err(Into::into)
    }

    fn try_get_tensor_mut(&mut self, id: usize) -> Result<&mut TensorHandle<T>> {
        self.tensors
            .get_mut(id)
            .and_then(|handle| handle.as_mut())
            .ok_or(StoreError::TensorNotFound { id })
            .map_err(Into::into)
    }
}

struct Builder<'a, D, T> {
    provider: &'a D,
    graph: &'a Graph<Definition>,
    initializers: Option<HashMap<usize, rmlk_schema::Tensor>>,
    arena: BufferArena,
    tensors: Vec<Option<TensorHandle<T>>>,
}

impl<'a, D, T> Builder<'a, D, T>
where
    D: DeviceService<Data = T>,
    T: DeviceData,
{
    fn new(
        provider: &'a D,
        graph: &'a Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Self {
        let arena = BufferArena::with_capacity(4096);
        let node_count = graph.node_count();
        let mut tensors = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            // TensorHandle does not implement clone so we cannot use the macro.
            tensors.push(None);
        }

        Self {
            provider,
            graph,
            initializers: Some(initializers),
            arena,
            tensors,
        }
    }

    fn load_initializers(&mut self) -> Result<()> {
        // Load initializers.
        let mut initializers = self
            .initializers
            .take()
            .expect("call load_initializers only once")
            .into_iter()
            .collect::<Vec<_>>();
        initializers.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
        for (node_id, ir_tensor) in initializers {
            debug!(
                "[node={node_id}][ir_tensor={}][dtype={:?}][shape={:?}]",
                ir_tensor.name.as_deref().unwrap_or(""),
                ir_tensor.data_type,
                ir_tensor.dims
            );

            debug_assert!(matches!(
                self.graph.get_node(node_id).map(|n| n.value().op()),
                // Todo: Fix this when we resolve the issue with Constants.
                Some(Op::Const) | Some(Op::NoOp)
            ));

            let data = match ir_tensor.data_type {
                DataType::Float16 => {
                    let on_host_data = f16::from_bytes(
                        ir_tensor
                            .raw_data
                            .as_ref()
                            .ok_or(StoreError::FailedToParseTensorRawData)?,
                    )?;
                    self.provider.htod_float16(on_host_data)?
                }
                DataType::Float => {
                    let on_host_data = match ir_tensor.float_data.is_empty() {
                        true => f32::from_bytes(
                            ir_tensor
                                .raw_data
                                .as_ref()
                                .ok_or(StoreError::FailedToParseTensorRawData)?,
                        )?,
                        false => {
                            // Todo: remove allocation.
                            ir_tensor.float_data
                        }
                    };
                    // let len = if on_host_data.len() >= 10 {
                    //     10
                    // } else {
                    //     on_host_data.len()
                    // };
                    //debug!("RAW_DATA (len={}) = <{:?}>", on_host_data.len(), &on_host_data[..len]);

                    self.provider.htod_float(on_host_data)?
                }
                DataType::Double => {
                    let on_host_data = match ir_tensor.double_data.is_empty() {
                        true => f64::from_bytes(
                            ir_tensor
                                .raw_data
                                .as_ref()
                                .ok_or(StoreError::FailedToParseTensorRawData)?,
                        )?,
                        false => {
                            // Todo: remove allocation.
                            ir_tensor.double_data
                        }
                    };
                    self.provider.htod_double(on_host_data)?
                }
                DataType::Int32 => {
                    let on_host_data = match ir_tensor.int32_data.is_empty() {
                        true => i32::from_bytes(
                            ir_tensor
                                .raw_data
                                .as_ref()
                                .ok_or(StoreError::FailedToParseTensorRawData)?,
                        )?,
                        false => {
                            // Todo: remove allocation.
                            ir_tensor.int32_data
                        }
                    };
                    self.provider.htod_i32(on_host_data)?
                }
                DataType::Int64 => {
                    let on_host_data = match ir_tensor.int64_data.is_empty() {
                        true => i64::from_bytes(
                            ir_tensor
                                .raw_data
                                .as_ref()
                                .ok_or(StoreError::FailedToParseTensorRawData)?,
                        )?,
                        false => {
                            // Todo: remove allocation.
                            ir_tensor.int64_data
                        }
                    };
                    let len = if on_host_data.len() >= 10 {
                        10
                    } else {
                        on_host_data.len()
                    };
                    debug!(
                        "RAW_DATA (len={}) = <{:?}>",
                        on_host_data.len(),
                        &on_host_data[..len]
                    );

                    self.provider.htod_i64(on_host_data)?
                }
                DataType::Bool => {
                    let on_host_data = match ir_tensor.bool_data.is_empty() {
                        true => bool::from_bytes(
                            ir_tensor
                                .raw_data
                                .as_ref()
                                .ok_or(StoreError::FailedToParseTensorRawData)?,
                        )?,
                        false => {
                            // Todo: remove allocation.
                            ir_tensor.bool_data
                        }
                    };
                    debug!("on_host_data = {:?}", on_host_data);

                    self.provider.htod_bool(on_host_data)?
                }
                _ => {
                    return Err(StoreError::DataTypeNotSupported {
                        dtype: ir_tensor.data_type,
                    }
                    .into())
                }
            };

            let arena_id = if ir_tensor.dims.is_empty() && data.len() == 1 {
                // Todo: handle scalars.
                ShapeBufArenaMut::new(&mut self.arena).alloc_from_shape_slice(&[])?
            } else {
                ShapeBufArenaMut::new(&mut self.arena)
                    .alloc_from_shape_slice(ir_tensor.dims.as_slice())?
            };

            let tensor = TensorHandle::new(
                ir_tensor.data_type,
                Some(arena_id),
                Rc::new(RefCell::new(Some(data))),
            );
            self.tensors[node_id].replace(tensor);
        }

        Ok(())
    }

    fn load_inputs(&mut self) -> Result<()> {
        for node_id in self.graph.inputs() {
            match self.graph.get_node(node_id) {
                Some(node) => {
                    let def = node.value();
                    let dtype = def
                        .dtype()
                        .ok_or(StoreError::DataTypeNotFound { node_id })?;
                    let shape = def.shape().ok_or(StoreError::ShapeNotFound { node_id })?;
                    let arena_id = ShapeBufArenaMut::new(&mut self.arena)
                        .alloc_from_shape_slice(shape.as_slice())?;
                    let tensor =
                        TensorHandle::new(dtype, Some(arena_id), Rc::new(RefCell::new(None)));
                    self.tensors[node_id].replace(tensor);
                }
                None => {
                    return Err(StoreError::UnknownInputNode { id: node_id }.into());
                }
            }
        }

        Ok(())
    }

    fn load_outputs(&mut self) -> Result<()> {
        for node_id in self.graph.outputs() {
            match self.graph.get_node(node_id) {
                Some(node) => {
                    let def = node.value();
                    let dtype = def
                        .dtype()
                        .ok_or(StoreError::DataTypeNotFound { node_id })?;
                    let shape = def.shape().ok_or(StoreError::ShapeNotFound { node_id })?;
                    let arena_id = ShapeBufArenaMut::new(&mut self.arena)
                        .alloc_from_shape_slice(shape.as_slice())?;
                    let tensor =
                        TensorHandle::new(dtype, Some(arena_id), Rc::new(RefCell::new(None)));
                    self.tensors[node_id].replace(tensor);
                }
                None => return Err(StoreError::UnknownOutputNode { id: node_id }.into()),
            }
        }

        Ok(())
    }

    fn load_op_outputs(&mut self) -> Result<()> {
        for (node_id, node) in self.graph.node_iter() {
            // We already loaded the initializers.
            if matches!(node.value().op(), Op::NoOp) {
                continue;
            }

            for output in node.outputs() {
                match self.graph.get_node(*output) {
                    Some(_) => {
                        if self
                            .tensors
                            .get(*output)
                            .ok_or(StoreError::OutputNodeNotFound {
                                node_id,
                                output_id: *output,
                            })?
                            .is_none()
                        {
                            self.tensors[*output].replace(TensorHandle::new(
                                DataType::Undefined,
                                None,
                                Rc::new(RefCell::new(None)),
                            ));
                        }
                    }
                    None => {
                        return Err(StoreError::OutputNodeNotFound {
                            output_id: *output,
                            node_id,
                        }
                        .into())
                    }
                }
            }
        }

        Ok(())
    }

    fn build(mut self) -> Result<TensorStore<T>> {
        self.load_inputs()?;
        self.load_outputs()?;
        self.load_op_outputs()?;
        self.load_initializers()?;

        Ok(TensorStore {
            tensors: self.tensors.into_boxed_slice(),
            buf_arena: self.arena,
        })
    }
}

#[derive(Debug)]
pub enum StoreError {
    TensorNotFound { id: usize },
    OutputNodeNotFound { output_id: usize, node_id: usize },
    UnknownOutputNode { id: usize },
    UnknownInputNode { id: usize },
    FailedToParseTensorRawData,
    DataTypeNotFound { node_id: usize },
    ShapeNotFound { node_id: usize },
    DataTypeNotSupported { dtype: DataType },
}

impl Display for StoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::TensorNotFound { id } => write!(f, "tensor not found: {}", id),
            StoreError::OutputNodeNotFound { output_id, node_id } => write!(
                f,
                "Output node `{output_id}` not found for node `{node_id}`"
            ),
            StoreError::UnknownOutputNode { id } => write!(f, "Unknown output node `{id}`"),
            StoreError::UnknownInputNode { id } => write!(f, "Unknown input node `{id}`"),
            StoreError::FailedToParseTensorRawData => write!(f, "failed to parse tensor raw data"),
            StoreError::DataTypeNotFound { node_id } => {
                write!(f, "DataType not found in node `{}`", node_id)
            }
            StoreError::ShapeNotFound { node_id } => {
                write!(f, "shape not found in node `{}`", node_id)
            }
            StoreError::DataTypeNotSupported { dtype } => {
                write!(f, "dataType `{:?}` not supported for tensor", dtype)
            }
        }
    }
}

impl std::error::Error for StoreError {}
