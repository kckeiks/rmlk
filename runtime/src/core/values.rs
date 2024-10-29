use crate::core::{DeviceService, Tensor};
use crate::{Error, Result};
use rmlk_graph::Graph;
use rmlk_schema::{DataType, Definition, Op};
use std::collections::HashMap;

/// Tensor values.
///
/// All the values for a computational graph.
pub struct Values<T> {
    inner: Box<[Option<Tensor<T>>]>,
}

impl<T> Values<T> {
    pub fn new<D: DeviceService<Data = T>>(
        provider: &D,
        graph: &Graph<Definition>,
        initializers: HashMap<usize, rmlk_schema::Tensor>,
    ) -> Result<Values<D::Data>> {
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
                true => to_float_vec(ir_tensor.raw_data.as_ref().ok_or(Error::MissingData)?),
                false => {
                    // Todo: remove allocation.
                    ir_tensor.float_data.clone()
                }
            };
            let data = provider
                .htod_float(data)
                .map_err(|_| Error::AllocationFailed)?;

            let mut tensor = Tensor::new_with_shape(ir_tensor.data_type, ir_tensor.dims.clone());
            tensor.init(data);
            tensors[node_id].replace(tensor);
        }

        for node_id in graph.inputs() {
            match graph.get_node(node_id) {
                Some(node) => {
                    let def = node.value();
                    // Todo: Throw an error instead.
                    let dtype = def.dtype().expect("Input should have a data type defined");
                    let tensor = if let Some(shape) = def.shape() {
                        // Todo: Remove clone.
                        Tensor::new_with_shape(dtype, shape.clone())
                    } else {
                        Tensor::new(dtype)
                    };
                    tensors[node_id].replace(tensor);
                }
                None => return Err(Error::MissingNode),
            }
        }

        for node_id in graph.outputs() {
            match graph.get_node(node_id) {
                Some(node) => {
                    let def = node.value();
                    // Todo: Throw an error instead.
                    let dtype = def.dtype().expect("Input should have a data type defined");
                    let tensor = if let Some(shape) = def.shape() {
                        // Todo: Remove clone.
                        Tensor::new_with_shape(dtype, shape.clone())
                    } else {
                        Tensor::new(dtype)
                    };
                    tensors[node_id].replace(tensor);
                }
                None => return Err(Error::MissingNode),
            }
        }

        for node in graph.nodes() {
            // We already loaded the initializers.
            if matches!(node.value().op(), Op::Const | Op::NoOp) {
                continue;
            }

            for output in node.outputs() {
                match graph.get_node(*output) {
                    Some(_) => {
                        if tensors.get(*output).ok_or(Error::MissingNode)?.is_none() {
                            tensors[*output].replace(Tensor::new(DataType::Undefined));
                        }
                    }
                    None => return Err(Error::MissingNode),
                }
            }
        }

        Ok(Values {
            inner: tensors.into_boxed_slice(),
        })
    }

    pub fn get(&self, id: usize) -> Option<&Tensor<T>> {
        self.inner.get(id).map(|r| r.as_ref()).flatten()
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut Tensor<T>> {
        self.inner.get_mut(id).map(|r| r.as_mut()).flatten()
    }
}

// Todo: Move to utils after refactor.
fn to_float_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
