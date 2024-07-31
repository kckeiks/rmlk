use crate::core::provider::Provider;
use rmlk_graph::Graph;
use std::sync::Arc;

pub struct SessionState {
    plan: Box<[usize]>,
    graph: Arc<Graph>,
    provider: Box<[Provider]>,
}

impl SessionState {
    pub fn new(plan: Box<[usize]>, graph: Graph, provider: Box<[Provider]>) -> Self {
        Self {
            plan,
            provider,
            graph: Arc::new(graph),
        }
    }

    pub fn providers(&self) -> impl Iterator<Item = &Provider> {
        self.provider.iter()
    }

    pub fn plan(&self) -> impl Iterator<Item = usize> + '_ {
        self.plan.iter().copied()
    }

    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }
}
