use crate::core::error::{Error, Result};
use crate::core::kernel::Kernel;
use crate::core::ops::flatten::FlattenOp;
use crate::core::provider::cuda::activation::ActivationKernel;
use crate::core::provider::cuda::conv::ConvKernel;
use crate::core::provider::cuda::data::CudaData;
use crate::core::provider::cuda::gemm::GemmKernel;
use crate::core::provider::cuda::global_average_pool::GlobalAveragePoolKernel;
use crate::core::provider::cuda::kernel::add::AddKernel;
use crate::core::provider::cuda::kernel::CudaKernel;
use crate::core::provider::cuda::max_pool::MaxPoolKernel;
use crate::core::provider::ExecutionProvider;
use crate::core::tensor;
use crate::core::tensor::Tensor;
use cudarc::driver::{CudaDevice, CudaFunction};
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Op};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub struct CudaProvider {
    device: Arc<CudaDevice>,
}

impl CudaProvider {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    fn load_kernel(&self, op: Op, dtype: DataType) -> Result<CudaFunction> {
        rmlk_cuda::load_kernel(&self.device, op, dtype).map_err(|_| Error::Unknown)
    }

    pub fn htod_float(&self, data: Vec<f32>) -> Result<CudaData> {
        let ptr = self
            .device
            .htod_copy(data)
            .map_err(|_| Error::AllocationFailed)?;
        Ok(CudaData::F32(ptr))
    }

    pub fn dtoh_float(&self, data: &mut CudaData) -> Result<Vec<f32>> {
        match data {
            CudaData::F32(ptr) => {
                let result = self
                    .device
                    .dtoh_sync_copy::<f32, _>(ptr)
                    .map_err(|_| Error::AllocationFailed)?;

                Ok(result)
            }
            _ => Err(Error::Unknown),
        }
    }
}

impl ExecutionProvider for CudaProvider {
    type Kernel = CudaKernel;

    fn allocate_execution_state(
        &mut self,
        graph: &Graph,
        plan: &[usize],
    ) -> Result<(
        HashMap<usize, usize>,
        Box<[Option<Tensor<<Self::Kernel as Kernel>::Data>>]>,
        Box<[usize]>,
    )> {
        // Todo: We might need the max id of the graph instead.
        let node_count = graph.nodes().count();
        let mut index_to_tensor_index = HashMap::new();

        let mut tensors = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            tensors.push(None);
        }

        let mut node_tensors = Vec::with_capacity(3 * graph.nodes().count());

        // Load initializers.
        for (node_id, ir_tensor) in graph.initializers() {
            debug_assert_eq!(graph.get_node(*node_id).map(|n| n.op()), Some(Op::Const));

            let data = to_float_vec(ir_tensor.raw_data.as_ref().ok_or(Error::MissingData)?);
            let ptr = self
                .device
                .htod_copy(data)
                .map_err(|_| Error::AllocationFailed)?;

            let mut tensor = Tensor::new_with_shape(ir_tensor.data_type, ir_tensor.dims.clone());
            tensor.init(CudaData::F32(ptr));
            tensors[*node_id].replace(tensor);
        }

        for node_id in graph.inputs() {
            match graph.get_node(node_id) {
                Some(node) => {
                    let def = node.def();
                    let tensor = if def.shape.is_empty() {
                        Tensor::new(def.dtype)
                    } else {
                        Tensor::new_with_shape(def.dtype, def.shape.clone())
                    };
                    tensors[node_id].replace(tensor);
                }
                None => return Err(Error::MissingNode),
            }
        }

        for node_id in graph.outputs() {
            match graph.get_node(node_id) {
                Some(node) => {
                    let def = node.def();
                    let tensor = if def.shape.is_empty() {
                        Tensor::new(def.dtype)
                    } else {
                        Tensor::new_with_shape(def.dtype, def.shape.clone())
                    };
                    tensors[node_id].replace(tensor);
                }
                None => return Err(Error::MissingNode),
            }
        }

        // Todo: Remove when we have a plan with steps to traverse the graph.
        for (node_id, node) in graph.nodes().enumerate() {
            // We already loaded the initializers.
            if graph.get_initial_tensor(node_id).is_some()
                || matches!(node.op(), Op::Const | Op::NoOp)
            {
                continue;
            }

            let index = node_tensors.len();
            index_to_tensor_index.insert(node_id, index);

            for input in node.inputs() {
                match graph.get_node(*input) {
                    Some(_) => {
                        debug_assert!(tensors.get(*input).map(Option::as_ref).flatten().is_some());

                        node_tensors.push(*input);
                    }
                    None => return Err(Error::MissingNode),
                }
            }

            for output in node.outputs() {
                match graph.get_node(*output) {
                    Some(_) => {
                        if tensors.get(*output).ok_or(Error::MissingNode)?.is_none() {
                            tensors[*output].replace(Tensor::new(DataType::Undefined));
                        }
                        node_tensors.push(*output);
                    }
                    None => return Err(Error::MissingNode),
                }
            }
        }

        println!("index_to_tensor_index={index_to_tensor_index:?}");
        println!("node_tensors={node_tensors:?}");
        Ok((
            index_to_tensor_index,
            tensors.into_boxed_slice(),
            node_tensors.into_boxed_slice(),
        ))
    }

    fn get_kernel(&self, op: Op, dtype: DataType) -> Result<Self::Kernel> {
        let kernel = match op {
            Op::Add => {
                // Todo: At what point should we load the kernel on device?
                let f = self.load_kernel(op, dtype)?;
                CudaKernel::Add(AddKernel::new(self.device.clone(), f))
            }
            Op::Gemm => CudaKernel::Gemm(GemmKernel::new(self.device.clone())),
            Op::Relu => CudaKernel::Relu(ActivationKernel::new(self.device.clone())),
            Op::Conv => CudaKernel::Conv(ConvKernel::new(self.device.clone())),
            Op::GlobalAveragePool => {
                CudaKernel::GlobalAveragePool(GlobalAveragePoolKernel::new(self.device.clone()))
            }
            Op::MaxPool => CudaKernel::MaxPool(MaxPoolKernel::new(self.device.clone())),
            Op::Flatten => CudaKernel::Flatten(FlattenOp::new()),
            _ => return Err(Error::NotSupported),
        };

        Ok(kernel)
    }
}

// Todo: Move to utils after refactor.
fn to_float_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
