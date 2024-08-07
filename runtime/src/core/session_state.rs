use crate::core::plan::Plan;
use crate::core::ExecutionProvider;
use rmlk_graph::Graph;
use std::sync::Arc;

pub struct ModelInstanceState<P> {
    graph: Arc<Graph>,
    _plan: Plan<P>,
}

impl<P> ModelInstanceState<P>
where
    P: ExecutionProvider,
{
    pub fn new(plan: Plan<P>, graph: Graph) -> Self {
        Self {
            _plan: plan,
            graph: Arc::new(graph),
        }
    }

    pub fn _plan(&self) -> &Plan<P> {
        &self._plan
    }

    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }
}
