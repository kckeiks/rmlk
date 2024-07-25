use ndarray::ArrayD;
use rmlk_graph::Graph;
use rmlk_ir::Model;
use std::collections::HashMap;
use std::sync::Arc;
use crate::core::error::{Error, Result};
use crate::core::kernel::KernelComputer;
use crate::core::provider::{ExecutionProvider, Provider};

// API: public. loads model from file or memory.
// An inference session.
// This contains session options.
pub struct Builder {
    graph: Graph,
}

impl Builder {
    pub fn with_model_from_memory(model: Box<[u8]>) -> Result<Self> {
        let model =
            bincode::deserialize::<Model>(&model).map_err(|_| Error::ModelDeserializationFailed)?;
        let graph = crate::parse::parse_ir_graph(model.graph.unwrap()).unwrap();

        Ok(Self { graph })
    }


    //
    // fn build_with_provider<P>(mut self, provider: P) -> Result<Session>
    // where
    //     P: ExecutionProvider + Clone + 'static,
    // {
    //     let (_, plan) =
    //         rmlk_graph::compute_order(self.graph.nodes_slice(), self.graph.outputs_slice())
    //             .map_err(|_| Error::ComputingPlanFailed)?;
    //
    //     provider.check_capacity(&mut self.graph, plan.as_slice());
    //
    //     // Todo: Provider returns a ExecutionState
    //     let session_state = Arc::new(SessionState {
    //         plan: plan.into_boxed_slice(),
    //         graph: Arc::new(self.graph),
    //         provider: Box::new([]),
    //     });
    //
    //     // let kernel_computer = KernelComputerStruct::new(&provider, &session_state);
    //
    //     Ok(Session { session_state, computers: Box::new([]) })
    // }
}

pub struct Session {
    session_state: Arc<SessionState>,
    computers: Box<[KernelComputer]>,
}

impl Session {
    // Todo: we must support other types.
    pub fn run(
        &mut self,
        input: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        // for step in self.session_state.plan.iter() {
        //     let node = self.session_state.graph.get_node(*step).ok_or(Error::MissingNode)?;
        //     let op = node.op();
        //
        // }
        //
        // Err(Error::ModelDeserializationFailed)
        todo!()
    }
}

pub struct SessionState {
    plan: Box<[usize]>,
    graph: Arc<Graph>,
    provider: Box<[Provider]>,
}
