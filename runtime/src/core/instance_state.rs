use crate::core::plan::Plan;
use crate::core::DeviceService;
use rmlk_graph::Graph;
use std::sync::Arc;

pub struct ModelInstanceState<D> {
    graph: Arc<Graph>,
    _plan: Plan<D>,
}

impl<D> ModelInstanceState<D>
where
    D: DeviceService,
{
    pub fn new(plan: Plan<D>, graph: Graph) -> Self {
        Self {
            _plan: plan,
            graph: Arc::new(graph),
        }
    }

    pub fn _plan(&self) -> &Plan<D> {
        &self._plan
    }

    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }
}
