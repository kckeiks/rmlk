use crate::core::device_service::{Value, ValueStore};
use crate::core::error::UnsupportedDataType;
use crate::providers::cuda::allocator::CudaBump;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::tensor::Tensor;
use crate::utils::{FromBytes, ShapeAllocator};
use anyhow::Result;
use cudarc::driver::DeviceRepr;
use half::f16;
use log::debug;
use rmlk_graph::Graph;
use rmlk_schema::{DataType, DataTypeMap, Definition, Op};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

pub struct TensorStore {
    scratch_alloc: Rc<CudaBump>,
    static_alloc: Rc<CudaBump>,
    tensors: Box<[Option<Tensor>]>,
    shape_alloc: ShapeAllocator,
}

impl TensorStore {
    pub fn new(
        scratch_alloc: Rc<CudaBump>,
        static_alloc: Rc<CudaBump>,
        shape_alloc: ShapeAllocator,
    ) -> TensorStore {
        Self {
            scratch_alloc,
            static_alloc,
            shape_alloc,
            tensors: Box::new([]),
        }
    }

    pub fn init(
        &mut self,
        graph: &Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Result<()> {
        let store = Builder::new(
            self.scratch_alloc.clone(),
            self.static_alloc.clone(),
            self.shape_alloc.clone(),
            graph,
            initializers,
        )
        .build()?;
        self.tensors = store.tensors;
        Ok(())
    }

    pub fn get(&self, id: usize) -> Option<Tensor> {
        self.tensors.get(id)?.clone()
    }
}

struct Builder<'a> {
    scratch_alloc: Rc<CudaBump>,
    static_alloc: Rc<CudaBump>,
    graph: &'a Graph<Definition>,
    initializers: Option<HashMap<usize, rmlk_schema::Tensor>>,
    shape_alloc: ShapeAllocator,
    tensors: Vec<Option<Tensor>>,
}

impl<'a> Builder<'a> {
    fn new(
        scratch_alloc: Rc<CudaBump>,
        static_alloc: Rc<CudaBump>,
        shape_alloc: ShapeAllocator,
        graph: &'a Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Self {
        let node_count = graph.node_count();
        let mut tensors = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            // TensorHandle does not implement clone so we cannot use the macro.
            tensors.push(None);
        }

        Self {
            scratch_alloc,
            static_alloc,
            graph,
            initializers: Some(initializers),
            shape_alloc,
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

            let tensor = Tensor::new(
                Rc::new(self.shape_alloc.empty()),
                Rc::new(RefCell::new(None)),
                self.static_alloc.clone(),
            );

            match ir_tensor.data_type {
                DataType::Float16 => {
                    let on_host_data = f16::from_bytes(
                        ir_tensor
                            .raw_data
                            .as_ref()
                            .ok_or(StoreError::FailedToParseTensorRawData)?,
                    )?;
                    // self.cuda_alloc
                    //     .alloc_from_slice::<f16>(&on_host_data)
                    //     ?
                    tensor.write_payload_from_slice::<f16>(&on_host_data)?;
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
                    // self.cuda_alloc
                    //     .alloc_from_slice::<f32>(&on_host_data)
                    //     ?
                    tensor.write_payload_from_slice::<f32>(&on_host_data)?;
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
                    // self.cuda_alloc
                    //     .alloc_from_slice::<f64>(&on_host_data)
                    //     ?
                    tensor.write_payload_from_slice::<f64>(&on_host_data)?;
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
                    // self.cuda_alloc
                    //     .alloc_from_slice::<i32>(&on_host_data)
                    //     ?
                    tensor.write_payload_from_slice::<i32>(&on_host_data)?;
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

                    // self.cuda_alloc
                    //     .alloc_from_slice::<i64>(&on_host_data)
                    //     ?
                    tensor.write_payload_from_slice::<i64>(&on_host_data)?;
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

                    // self.cuda_alloc
                    //     .alloc_from_slice::<bool>(&on_host_data)
                    //     ?
                    tensor.write_payload_from_slice::<bool>(&on_host_data)?;
                }
                _ => {
                    return Err(StoreError::DataTypeNotSupported {
                        dtype: ir_tensor.data_type,
                    }
                    .into())
                }
            };

            if ir_tensor.dims.is_empty() && tensor.len() == 1 {
                // Todo: handle scalars.
                tensor.copy_shape_from_slice(&[]);
                //self.slab_alloc.alloc_from_slice(&[])
            } else {
                tensor.copy_shape_from_slice(ir_tensor.dims.as_slice());
                //self.slab_alloc.alloc_from_slice(ir_tensor.dims.as_slice())
            };

            self.tensors[node_id].replace(tensor);
        }

        Ok(())
    }

    fn load_inputs(&mut self) -> Result<()> {
        for node_id in self.graph.inputs() {
            match self.graph.get_node(node_id) {
                Some(node) => {
                    let def = node.value();
                    let shape = def.shape().ok_or(StoreError::ShapeNotFound { node_id })?;
                    let arena_id = self.shape_alloc.alloc_from_slice(shape.as_slice());
                    let tensor = Tensor::new(
                        Rc::new(arena_id),
                        Rc::new(RefCell::new(None)),
                        self.scratch_alloc.clone(),
                    );
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
                    let shape = def.shape().ok_or(StoreError::ShapeNotFound { node_id })?;
                    let arena_id = self.shape_alloc.alloc_from_slice(shape.as_slice());
                    let tensor = Tensor::new(
                        Rc::new(arena_id),
                        Rc::new(RefCell::new(None)),
                        self.scratch_alloc.clone(),
                    );
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
                            self.tensors[*output].replace(Tensor::new(
                                Rc::new(self.shape_alloc.empty()),
                                Rc::new(RefCell::new(None)),
                                self.scratch_alloc.clone(),
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

    fn build(mut self) -> Result<TensorStore> {
        self.load_inputs()?;
        self.load_outputs()?;
        self.load_op_outputs()?;
        self.load_initializers()?;

        Ok(TensorStore {
            scratch_alloc: self.scratch_alloc,
            static_alloc: self.static_alloc,
            tensors: self.tensors.into_boxed_slice(),
            shape_alloc: self.shape_alloc,
        })
    }
}

impl ValueStore for TensorStore {
    type Value = Tensor;

    fn init(
        &mut self,
        graph: &Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Result<()> {
        self.init(graph, initializers)
    }

    fn get(&self, id: usize) -> Option<Self::Value> {
        self.get(id)
    }

    fn clear(&self) {
        self.scratch_alloc.clear();
    }
}

impl Value for Tensor {
    type Data = CudaData;

    fn set_data(&mut self, data: Self::Data) -> Result<()> {
        let dtype = data.dtype();
        match dtype {
            DataType::Float16 => {
                let src = data.data::<f16>();
                self.write_payload(&src)?;
            }
            DataType::Float => {
                let src = data.data::<f32>();
                self.write_payload(&src)?;
            }
            DataType::Double => {
                let src = data.data::<f64>();
                self.write_payload(&src)?;
            }
            DataType::Int32 => {
                let src = data.data::<i32>();
                self.write_payload(&src)?;
            }
            DataType::Int64 => {
                let src = data.data::<i64>();
                self.write_payload(&src)?;
            }
            DataType::Bool => {
                let src = data.data::<bool>();
                self.write_payload(&src)?;
            }
            dtype => return Err(UnsupportedDataType(dtype).into()),
        }

        Ok(())
    }

    fn set_shape(&mut self, shape: &[usize]) -> Result<()> {
        self.copy_shape_from_slice(shape);
        Ok(())
    }

    fn data<T>(&self) -> Result<Vec<T>>
    where
        T: DataTypeMap + DeviceRepr + Default + Clone,
    {
        self.payload_to_vec()
    }

    fn shape(&self) -> Vec<usize> {
        Tensor::shape(self).to_vec()
    }

    fn dtype(&self) -> DataType {
        Tensor::dtype(self)
    }
}

#[derive(Debug)]
pub enum StoreError {
    OutputNodeNotFound { output_id: usize, node_id: usize },
    UnknownOutputNode { id: usize },
    UnknownInputNode { id: usize },
    FailedToParseTensorRawData,
    ShapeNotFound { node_id: usize },
    DataTypeNotSupported { dtype: DataType },
}

impl Display for StoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::OutputNodeNotFound { output_id, node_id } => write!(
                f,
                "Output node `{output_id}` not found for node `{node_id}`"
            ),
            StoreError::UnknownOutputNode { id } => write!(f, "Unknown output node `{id}`"),
            StoreError::UnknownInputNode { id } => write!(f, "Unknown input node `{id}`"),
            StoreError::FailedToParseTensorRawData => write!(f, "failed to parse tensor raw data"),
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
