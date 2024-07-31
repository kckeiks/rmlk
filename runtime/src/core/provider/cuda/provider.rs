use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::kernel::Kernel;
use crate::core::provider::cuda::activation::ActivationKernel;
use crate::core::provider::cuda::conv::ConvKernel;
use crate::core::provider::cuda::data::CudaData;
use crate::core::provider::cuda::gemm::GemmKernel;
use crate::core::provider::cuda::global_average_pool::GlobalAveragePoolKernel;
use crate::core::provider::cuda::kernel::add::AddKernel;
use crate::core::provider::cuda::kernel::CudaKernel;
use crate::core::provider::cuda::max_pool::MaxPoolKernel;
use crate::core::provider::ExecutionProvider;
use crate::core::tensor::Tensor;
use cudarc::driver::{CudaDevice, CudaFunction};
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Op};
use std::collections::HashMap;
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
}

impl ExecutionProvider for CudaProvider {
    type Kernel = CudaKernel;

    fn allocate_execution_state(
        &mut self,
        graph: &Graph,
        plan: &[usize],
    ) -> Result<(
        HashMap<usize, usize>,
        Box<[Tensor<<Self::Kernel as Kernel>::Data>]>,
        Box<[usize]>,
    )> {
        let node_count = graph.nodes().count();
        let mut index_to_tensor_index = HashMap::new();

        let mut tensors = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            tensors.push(Tensor::new(DataType::Undefined));
        }

        let mut node_tensors = Vec::with_capacity(3 * graph.nodes().count());

        // Todo: Remove these enumerates.
        for (node_id, node) in plan.iter().enumerate() {
            let node = graph.get_node(*node).ok_or(Error::MissingNode)?;

            if !node.inputs().is_empty() {
                let index = node_tensors.len();
                index_to_tensor_index.insert(node_id, index);
            }

            for input in node.inputs() {
                match graph.get_node(*input) {
                    Some(node) => {
                        if let Op::Const = node.op() {
                            if tensors
                                .get(*input)
                                .map(|t| t.data().is_none())
                                .ok_or(Error::MissingData)?
                            {
                                // One initializer should only have one node.
                                debug_assert!(node.inputs().len() == 1);

                                let input = node.inputs().get(0).ok_or(Error::MissingData)?;
                                let initializer =
                                    graph.get_initial_tensor(*input).ok_or(Error::MissingData)?;
                                let ptr = self
                                    .device
                                    .htod_copy(initializer.float_data.clone())
                                    .map_err(|_| Error::AllocationFailed)?;

                                let mut tensor = Tensor::new_with_shape(
                                    DataType::Float,
                                    initializer.dims.clone(),
                                );
                                tensor.init(CudaData::F32(ptr));
                                tensors[*input] = tensor;
                            }
                        }

                        node_tensors.push(*input);
                    }
                    None => return Err(Error::MissingNode),
                }
            }

            for output in node.outputs() {
                match graph.get_node(*output) {
                    Some(node) => {
                        if let Op::Const = node.op() {
                            if tensors
                                .get(*output)
                                .map(|t| t.data().is_none())
                                .ok_or(Error::MissingData)?
                            {
                                // One initializer should only have one node.
                                debug_assert!(node.inputs().len() == 1);

                                let input = node.inputs().get(0).ok_or(Error::MissingData)?;
                                let initializer =
                                    graph.get_initial_tensor(*input).ok_or(Error::MissingData)?;
                                let ptr = self
                                    .device
                                    .htod_copy(initializer.float_data.clone())
                                    .map_err(|_| Error::AllocationFailed)?;

                                let mut tensor = Tensor::new_with_shape(
                                    DataType::Float,
                                    initializer.dims.clone(),
                                );
                                tensor.init(CudaData::F32(ptr));
                                tensors[*input] = tensor;
                            }
                        }

                        node_tensors.push(*output);
                    }
                    None => return Err(Error::MissingNode),
                }
            }
        }

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
            Op::Flatten => {
                todo!()
            }
            _ => return Err(Error::NotSupported),
        };

        Ok(kernel)
    }
}

pub struct CudaAllocator {
    inner: Arc<CudaDevice>,
}
