use std::collections::HashMap;
use rmlk_graph::Graph;
use std::sync::Arc;
use ndarray::ArrayD;
use rmlk_ir::Model;
use rmlk_tensor::{Context, CudaData, ExecutionState};
use crate::error::{Error, Result};


// API: public. loads model from file or memory.
// An inference session.
// This contains session options.
pub struct Builder {
    graph: Arc<Graph>,
    state: SessionState,
}

impl Builder {
    pub fn from_model(model: Box<[u8]>) -> Result<Self> {
        let model = bincode::deserialize::<Model>(&model).map_err(|_| Error::ModelDeserializationFailed)?;
        let graph: Graph = model.graph.try_into().map_err(|_| Error::ModelDeserializationFailed)?;
        let graph = Arc::new(graph);

        Ok(Self {
            state: SessionState::new(graph.clone()),
            graph,
        })
    }

    pub fn load_model_on_cuda(self) -> Result<Session> {
        let (_, plan) = rmlk_graph::compute_order(self.graph.nodes(), self.graph.outputs()).map_err(|_| Error::ComputingPlanFailed)?;

        // Todo: create plan object.
        // - Traverse steps and allocate tensors headers.
        // - Allocate decide data for initializers.

        Ok(Session {
            graph: self.graph,
            state: self.state,
        })
    }
}

pub struct Session {
    graph: Arc<Graph>,
    state: SessionState,
}

impl Session {
    // Todo: we must support other types.
    pub fn run(input: HashMap<String, ArrayD<f32>>) -> Result<HashMap<String, ArrayD<f32>>> {
        todo!()
    }
}


pub struct SessionState {
    plan: Plan,
    graph: Arc<Graph>,
    execution_state: Box<[ProviderExecutionState]>,
    providers: Box<[Box<dyn ExecutionProvider>]>,
}

enum ProviderExecutionState {
    Cuda(Option<ExecutionState<CudaData>>),
}

pub trait ExecutionProvider {

}

impl SessionState {
    fn new(graph: Arc<Graph>, plan: Plan, execution_state: ExecutionState<T>) -> Self {
        Self {
            plan,
            graph,
            execution_state,
            providers: Box::new([Box::new(CudaProvider::new())]),
        }
    }
}

pub struct Plan {
    steps: Box<usize>,
}

pub struct CudaProvider {

}

impl CudaProvider {
    pub fn new() -> Self {
        todo!()
    }
}
