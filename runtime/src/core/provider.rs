use std::collections::HashMap;
use rmlk_graph::Graph;
use rmlk_ir::Op;
use rmlk_tensor::CudaKernel;

pub trait ExecutionProvider {
    fn check_capacity(&self, graph: &mut Graph, plan: &[usize]) -> Option<Graph>;
    fn registry(&self) -> Registry;
}

pub enum Provider {
    Cuda,
    Cpu,
}

pub enum Registry {
    Cuda(HashMap<Op, CudaKernel>),
    Cpu(HashMap<Op, CudaKernel>),
}

