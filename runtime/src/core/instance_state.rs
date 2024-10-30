use crate::core::plan::Plan;
use crate::core::DeviceService;
use rmlk_graph::Graph;
use rmlk_schema::Definition;
use std::collections::HashMap;
use std::sync::Arc;

/// The state of the model instance.
///
/// This object is used internally by the runtime to
/// hold the state of its corresponding model instance.
pub struct ModelInstanceState<D> {
    graph: Arc<Graph<Definition>>,
    map_io_name_to_id: HashMap<String, usize>,
    _plan: Plan<D>,
}

impl<D> ModelInstanceState<D>
where
    D: DeviceService,
{
    pub fn new(
        plan: Plan<D>,
        graph: Graph<Definition>,
        map_io_name_to_id: HashMap<String, usize>,
    ) -> Self {
        Self {
            map_io_name_to_id,
            _plan: plan,
            graph: Arc::new(graph),
        }
    }

    pub fn _plan(&self) -> &Plan<D> {
        &self._plan
    }

    pub fn graph(&self) -> &Arc<Graph<Definition>> {
        &self.graph
    }

    pub fn get_io_node_id(&self, name: &String) -> Option<usize> {
        self.map_io_name_to_id.get(name).copied()
    }
}
