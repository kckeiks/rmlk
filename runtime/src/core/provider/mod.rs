pub mod cuda;

use cuda::CudaProvider;
use rmlk_graph::Graph;
use rmlk_ir::Op;
use rmlk_tensor::cuda::CudaKernel;
use std::collections::HashMap;

type Result<T> = std::result::Result<T, Error>;

enum Error {
    Unknown,
}

pub trait ExecutionProvider {
    fn check_capacity(&self, graph: &mut Graph, plan: &[usize]) -> Option<Graph>;
    fn registry(&self) -> Registry;
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
    Cuda(HashMap<Op, CudaKernel>),
    Cpu(HashMap<Op, CudaKernel>),
}
