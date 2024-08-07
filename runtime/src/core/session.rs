use crate::core::context::Context;
use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::kernel::Kernel;
use crate::core::plan::Plan;
use crate::core::provider::ExecutionProvider;
use crate::core::session_state::ModelInstanceState;
use crate::providers::cuda::CudaProvider;
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

    pub fn build(self) -> Result<Session<CudaProvider>> {
        // Check for cycles and return ids of nodes that
        // are required for computing the outputs.
        let (_, plan) =
            rmlk_graph::compute_order(self.graph.nodes_slice(), self.graph.outputs_slice())
                .map_err(|_| Error::ComputingPlanFailed)?;

        let mut provider = CudaProvider::new(CudaDevice::new(0).map_err(|_| Error::Unknown)?);

        let graph = self.graph;
        let (node_to_tensor_set_index, tensors, node_tensors) =
            provider.allocate_execution_state(&graph, &plan)?;

        let plan = Plan::new(Box::new([provider]));

        let session_state = Arc::new(ModelInstanceState::new(plan, graph));

        Ok(Session {
            execution_state: Box::new([ExecutionState::new(
                session_state.clone(),
                tensors,
                node_tensors,
            )]),
            session_state,
            node_tensor_index_map: node_to_tensor_set_index,
        })
    }
}

pub struct Session<P: ExecutionProvider> {
    session_state: Arc<ModelInstanceState<P>>,
    execution_state: Box<[ExecutionState<P>]>,
    node_tensor_index_map: HashMap<usize, usize>,
}

impl<P> Session<P>
where
    P: ExecutionProvider,
{
    fn load_input(&mut self, data: Vec<f32>) -> Result<()> {
        let mut inputs = self.session_state.graph().inputs();

        let provider = self.session_state._plan().provider(0).unwrap();

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
                    .get_mut(0)
                    .expect("Provider is hardcoded")
                    .get_value(input)
                    .ok_or(Error::MissingData)
                    .unwrap();
                tensor.init(provider.htod_float(data)?);

                Ok(())
            }
        }
    }

    fn load_output(&mut self) -> Result<Vec<Vec<f32>>> {
        let provider = self.session_state._plan().provider(0).unwrap();

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
                        .get_mut(0)
                        .expect("Provider is hardcoded")
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

        let provider = self.session_state._plan().provider(0).unwrap();

        // Remove allocation.
        for (i, node) in self.session_state.graph().nodes_slice().iter().enumerate() {
            // let node = graph.get_node(i).ok_or(Error::MissingNode)?;
            // Todo: We might want to separate the load operation because at this point we don't know the type.
            if matches!(node.op(), Op::NoOp) || matches!(node.op(), Op::Const) {
                continue;
            }
            let kernel = provider.get_kernel(node.op(), DataType::Float)?;

            let mut ctx = Context::new(
                self.execution_state.get_mut(0).ok_or(Error::MissingData)?,
                &self.node_tensor_index_map,
                i,
            )
            .unwrap();
            trace!(
                "{i} {:?} {:?} inputs={:?}",
                node.op(),
                node.def().node.as_ref().unwrap().name,
                node.inputs()
            );
            kernel.compute(&mut ctx)?;
        }

        Ok(self.load_output().unwrap())
        // Ok(Vec::new())
    }
}
