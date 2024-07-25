use rmlk_graph::Graph;
use rmlk_ir::Op;
use rmlk_tensor::CudaKernel;
use std::collections::HashMap;
use std::marker::PhantomData;

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

pub struct CudaProvider {
    marker: PhantomData<()>,
}

impl CudaProvider {
    pub fn new() -> Self {
        todo!()
    }
}

impl ExecutionProvider for CudaProvider {
    fn check_capacity(&self, graph: &mut Graph, plan: &[usize]) -> Option<Graph> {
        todo!()
    }

    fn registry(&self) -> Registry {
        todo!()
    }
}

pub enum Registry {
    Cuda(HashMap<Op, CudaKernel>),
    Cpu(HashMap<Op, CudaKernel>),
}
