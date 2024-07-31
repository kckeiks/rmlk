pub mod cuda;

use crate::core::error::Result;
use crate::core::kernel::Kernel;
use crate::core::tensor::Tensor;
use cuda::CudaProvider;
use ndarray::Data;
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Op};
use std::collections::HashMap;

pub trait ExecutionProvider {
    type Kernel: Kernel;
    fn allocate_execution_state(
        &mut self,
        graph: &Graph,
        plan: &[usize],
    ) -> Result<(
        HashMap<usize, usize>,
        Box<[Tensor<<Self::Kernel as Kernel>::Data>]>,
        Box<[usize]>,
    )>;
    fn get_kernel(&self, op: Op, dtype: DataType) -> Result<Self::Kernel>;
}

pub enum Provider {
    Cuda(CudaProvider),
    Cpu,
}

impl Provider {
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }
}

pub enum Registry {
    Cuda(HashMap<Op, String>),
    Cpu(HashMap<Op, String>),
}
