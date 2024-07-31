use crate::core::context::Context;
use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::kernel::Kernel;
use crate::core::provider::cuda::{CudaExecutionState, CudaProvider};
use crate::core::provider::{ExecutionProvider, Provider};
use crate::core::session_state::SessionState;
use cudarc::driver::CudaDevice;
use ndarray::ArrayD;
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Model};
use std::collections::HashMap;
use std::sync::Arc;

pub const CPU_PROVIDER_ID: usize = 0;
pub const CUDA_PROVIDER_ID: usize = 1;

// API: public. loads model from file or memory.
// An inference session.
// This contains session options.
pub struct Builder {
    graph: Graph,
}

impl Builder {
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub fn with_model_from_memory(model: Box<[u8]>) -> Result<Self> {
        let model =
            bincode::deserialize::<Model>(&model).map_err(|_| Error::ModelDeserializationFailed)?;
        let graph = crate::parse::parse_ir_graph(model.graph.unwrap()).unwrap();

        Ok(Self { graph })
    }

    pub fn build(self) -> Result<Session> {
        // Check for cycles and return ids of nodes that
        // are required for computing the outputs.
        let (_, plan) =
            rmlk_graph::compute_order(self.graph.nodes_slice(), self.graph.outputs_slice())
                .map_err(|_| Error::ComputingPlanFailed)?;

        let mut provider = CudaProvider::new(CudaDevice::new(0).map_err(|_| Error::Unknown)?);

        let graph = self.graph;
        let (node_to_tensor_set_index, tensors, node_tensors) =
            provider.allocate_execution_state(&graph, &plan)?;

        let session_state = Arc::new(SessionState::new(
            plan.into_boxed_slice(),
            graph,
            Box::new([Provider::Cuda(provider)]),
        ));

        Ok(Session {
            execution_state: ExecutionState::new(session_state.clone(), tensors, node_tensors),
            session_state,
            node_tensor_index_map: node_to_tensor_set_index,
        })
    }
}

pub struct Session {
    session_state: Arc<SessionState>,
    execution_state: CudaExecutionState,
    node_tensor_index_map: HashMap<usize, usize>,
}

impl Session {
    pub fn run(
        &mut self,
        _input: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        let graph = self.session_state.graph().clone();
        for provider in self.session_state.providers() {
            match provider {
                Provider::Cuda(provider) => {
                    for i in self.session_state.plan() {
                        let node = graph.get_node(i).ok_or(Error::MissingNode)?;
                        // Todo: We might want to separate the load operation because at this point we don't know the type.
                        let kernel = provider.get_kernel(node.op(), DataType::Float)?;
                        let node_tensor_index = self
                            .node_tensor_index_map
                            .get(&i)
                            .ok_or(Error::MissingData)?;
                        let mut ctx = Context::new(&mut self.execution_state, *node_tensor_index)?;
                        kernel.compute(&mut ctx)?;
                    }
                }
                Provider::Cpu => {
                    return Err(Error::NotSupported);
                }
            }
        }

        Ok(HashMap::new())
    }
}
