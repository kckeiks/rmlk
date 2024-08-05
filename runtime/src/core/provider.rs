use crate::core::kernel::Kernel;
use crate::core::tensor::Tensor;
use crate::provider::cuda::CudaProvider;
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Op};
use std::collections::HashMap;

pub trait ExecutionProvider {
    type Kernel: Kernel;
    fn allocate_execution_state(
        &mut self,
        graph: &Graph,
        plan: &[usize],
    ) -> crate::Result<(
        HashMap<usize, usize>,
        Box<[Option<Tensor<<Self::Kernel as Kernel>::Data>>]>,
        Box<[usize]>,
    )>;
    fn get_kernel(&self, op: Op, dtype: DataType) -> crate::Result<Self::Kernel>;
}

#[allow(unused)]
pub enum Provider {
    Cuda(CudaProvider),
    Cpu,
}

impl Provider {
    pub fn _is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn _is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }
}
