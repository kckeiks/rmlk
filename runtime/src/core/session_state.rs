use crate::core::provider::Provider;
use rmlk_graph::Graph;
use std::sync::Arc;

pub struct SessionState {
    _plan: Box<[usize]>,
    graph: Arc<Graph>,
    provider: Box<[Provider]>,
}

impl SessionState {
    pub fn new(plan: Box<[usize]>, graph: Graph, provider: Box<[Provider]>) -> Self {
        Self {
            _plan: plan,
            provider,
            graph: Arc::new(graph),
        }
    }

    pub fn providers(&self) -> impl Iterator<Item = &Provider> {
        self.provider.iter()
    }

    pub fn _plan(&self) -> impl Iterator<Item = usize> + '_ {
        self._plan.iter().copied()
    }

    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }
}
