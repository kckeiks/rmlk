use crate::core::error::{Error, Result};
use crate::core::kernel::{CudaComputer, KernelComputer, ProviderComputer};
use crate::core::provider::{CudaProvider, ExecutionProvider, Provider};
use ndarray::ArrayD;
use rmlk_graph::Graph;
use rmlk_ir::Model;
use std::collections::HashMap;
use std::sync::Arc;

pub const CPU_PROVIDER_ID: usize = 0;
pub const CUDA_PROVIDER_ID: usize = 1;

// API: public. loads model from file or memory.
// An inference session.
// This contains session options.
pub struct Builder {
    graph: Graph,
    providers: [Option<Provider>; 2],
}

impl Builder {
    pub fn with_model_from_memory(model: Box<[u8]>) -> Result<Self> {
        let model =
            bincode::deserialize::<Model>(&model).map_err(|_| Error::ModelDeserializationFailed)?;
        let graph = crate::parse::parse_ir_graph(model.graph.unwrap()).unwrap();

        Ok(Self {
            graph,
            providers: [None, None],
        })
    }

    pub fn with_cuda_provider(self) -> Self {
        let mut providers = self.providers;
        providers[CUDA_PROVIDER_ID].replace(Provider::Cuda(CudaProvider::new()));

        Self {
            graph: self.graph,
            providers,
        }
    }

    pub fn build(self) -> Result<Session> {
        // Check for cycles and return ids of nodes that
        // are required for computing the outputs.
        let (_, plan) =
            rmlk_graph::compute_order(self.graph.nodes_slice(), self.graph.outputs_slice())
                .map_err(|_| Error::ComputingPlanFailed)?;

        let mut providers = self.providers;
        // In the future we can provide a default CPU provider instead of returning an error.
        let provider = providers[CUDA_PROVIDER_ID]
            .take()
            .ok_or(Error::NotSupported)?;

        let mut graph = self.graph;
        match &provider {
            Provider::Cuda(provider) => {
                provider.check_capacity(&mut graph, plan.as_slice());
            }
            Provider::Cpu => {
                return Err(Error::NotSupported);
            }
        }

        let is_cuda = provider.is_cuda();

        let session_state = Arc::new(SessionState {
            plan: plan.into_boxed_slice(),
            graph: Arc::new(graph),
            provider: Box::new([provider]),
        });

        let computers = match is_cuda {
            true => ProviderComputer::Cuda(CudaComputer::new(&session_state)),
            false => {
                return Err(Error::NotSupported);
            }
        };

        Ok(Session {
            session_state,
            computers: Box::new([computers]),
        })
    }
}

pub struct Session {
    session_state: Arc<SessionState>,
    computers: Box<[ProviderComputer]>,
}

impl Session {
    pub fn run(
        &mut self,
        input: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        todo!()
    }
}

pub struct SessionState {
    plan: Box<[usize]>,
    graph: Arc<Graph>,
    provider: Box<[Provider]>,
}

impl SessionState {
    pub fn new(plan: Box<[usize]>, graph: Graph, provider: Box<[Provider]>) -> Self {
        todo!()
    }
}
