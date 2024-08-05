use crate::core::context::Context;
use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::kernel::Kernel;
use crate::core::provider::{ExecutionProvider, Provider};
use crate::core::session_state::SessionState;
use crate::provider::cuda::{CudaExecutionState, CudaProvider};
use cudarc::driver::CudaDevice;
use log::trace;
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Model, Op};
use std::collections::HashMap;
use std::sync::Arc;

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
    fn load_input(&mut self, data: Vec<f32>) -> Result<()> {
        let mut inputs = self.session_state.graph().inputs();

        let provider = match self.session_state.providers().next().unwrap() {
            Provider::Cuda(provider) => provider,
            _ => return Err(Error::NotSupported),
        };

        let input = inputs.next().ok_or(Error::MissingData)?;
        match self.session_state.graph().get_node(input) {
            None => Err(Error::MissingData),
            Some(_) => {
                // let node_tensor_index = self
                //     .node_tensor_index_map
                //     .get(&input)
                //     .ok_or(Error::MissingData).unwrap();

                // Todo: Improve API for loading input values.
                let tensor = self
                    .execution_state
                    .get_value(input)
                    .ok_or(Error::MissingData)
                    .unwrap();
                tensor.init(provider.htod_float(data)?);

                Ok(())
            }
        }
    }

    fn load_output(&mut self) -> Result<Vec<Vec<f32>>> {
        let provider = match self.session_state.providers().next().unwrap() {
            Provider::Cuda(provider) => provider,
            _ => return Err(Error::NotSupported),
        };

        let mut result = Vec::new();
        for output in self.session_state.graph().outputs() {
            match self.session_state.graph().get_node(output) {
                None => return Err(Error::MissingData),
                Some(_) => {
                    // let node_tensor_index = self
                    //     .node_tensor_index_map
                    //     .get(&output)
                    //     .ok_or(Error::MissingData)?;

                    // Todo: Improve API for loading input values.
                    let tensor = self
                        .execution_state
                        .get_value(output)
                        .ok_or(Error::MissingData)?;
                    let ptr = tensor.data_mut().take().ok_or(Error::MissingData)?;
                    let data = provider.dtoh_float(ptr)?;
                    result.push(data);
                }
            }
        }

        Ok(result)
    }

    pub fn run(&mut self, input: Vec<f32>) -> Result<Vec<Vec<f32>>> {
        self.load_input(input).unwrap();

        for provider in self.session_state.providers() {
            match provider {
                Provider::Cuda(provider) => {
                    // Remove allocation.
                    for (i, node) in self.session_state.graph().nodes_slice().iter().enumerate() {
                        // let node = graph.get_node(i).ok_or(Error::MissingNode)?;
                        // Todo: We might want to separate the load operation because at this point we don't know the type.
                        if matches!(node.op(), Op::NoOp) || matches!(node.op(), Op::Const) {
                            continue;
                        }
                        let kernel = provider.get_kernel(node.op(), DataType::Float)?;

                        let mut ctx =
                            Context::new(&mut self.execution_state, &self.node_tensor_index_map, i)
                                .unwrap();
                        trace!(
                            "{i} {:?} {:?} inputs={:?}",
                            node.op(),
                            node.def().node.as_ref().unwrap().name,
                            node.inputs()
                        );
                        kernel.compute(&mut ctx)?;
                    }
                }
                Provider::Cpu => {
                    return Err(Error::NotSupported);
                }
            }
        }

        Ok(self.load_output().unwrap())
        // Ok(Vec::new())
    }
}
