use crate::provider::provider::Provider;
use rmlk_graph::Graph;
use std::sync::Arc;

pub struct OpKernelContext<P> {
    pub provider: P,
    pub inputs: Vec<usize>,
    pub graph_view: Arc<Graph>,
}

pub trait OpKernel {
    type Provider: Provider;
    fn compute(&self, ctx: OpKernelContext<Self::Provider>);
}
