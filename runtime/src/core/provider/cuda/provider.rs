use crate::core::provider::{ExecutionProvider, Registry};
use rmlk_graph::Graph;
use std::marker::PhantomData;

pub struct CudaProvider {
    marker: PhantomData<()>,
}

impl CudaProvider {
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl ExecutionProvider for CudaProvider {
    fn check_capacity(&self, _graph: &mut Graph, _plan: &[usize]) -> Option<Graph> {
        None
    }

    fn registry(&self) -> Registry {
        todo!()
    }
}
