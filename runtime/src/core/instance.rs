use crate::core::context::Context;
use crate::core::device_service::DeviceService;
use crate::core::error::{Error, Result};
use crate::core::execution_state::ExecutionState;
use crate::core::instance_state::ModelInstanceState;
use crate::core::kernel::Kernel;
use crate::core::plan::Plan;
use crate::core::values::Values;
use crate::providers::cuda::Cuda;
use cudarc::driver::CudaDevice;
use log::trace;
use rmlk_graph::Graph;
use rmlk_ir::{DataType, Model, Op};
use std::sync::Arc;

/// Model instance builder.
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

    pub fn build(self) -> Result<ModelInstance<Cuda>> {
        // Check for cycles and return ids of nodes that
        // are required for computing the outputs.
        // Todo: Implement plan.
        let (_, _plan) =
            rmlk_graph::compute_order(self.graph.nodes_slice(), self.graph.outputs_slice())
                .map_err(|_| Error::ComputingPlanFailed)?;

        let provider = Cuda::new(CudaDevice::new(0)?);
        let values = Values::new(&provider, &self.graph)?;
        let plan = Plan::new(Box::new([provider]));
        let instance_state = Arc::new(ModelInstanceState::new(plan, self.graph));

        Ok(ModelInstance {
            execution_state: Box::new([ExecutionState::new(instance_state.clone(), values)?]),
            session_state: instance_state,
        })
    }
}

/// A model instance.
///
/// This object represents the model instantiated in the
/// runtime.
pub struct ModelInstance<D: DeviceService> {
    session_state: Arc<ModelInstanceState<D>>,
    execution_state: Box<[ExecutionState<D>]>,
}

impl<D> ModelInstance<D>
where
    D: DeviceService,
{
    fn load_input(&mut self, data: Vec<f32>) -> Result<()> {
        let mut inputs = self.session_state.graph().inputs();

        let provider = self
            .session_state
            ._plan()
            .device(0)
            .expect("We always have one device");
        let input = inputs.next().ok_or(Error::MissingData)?;
        match self.session_state.graph().get_node(input) {
            None => Err(Error::MissingData),
            Some(_) => {
                // Todo: Improve API for loading input values.
                let tensor = self
                    .execution_state
                    .get_mut(0)
                    .expect("Provider is hardcoded")
                    .get_value_from_node_id_mut(input)
                    .ok_or(Error::MissingData)?;
                tensor.init(provider.htod_float(data)?);

                Ok(())
            }
        }
    }

    fn load_output(&mut self) -> Result<Vec<Vec<f32>>> {
        let provider = self
            .session_state
            ._plan()
            .device(0)
            .expect("We always have one device");

        let mut result = Vec::new();
        for output in self.session_state.graph().outputs() {
            match self.session_state.graph().get_node(output) {
                None => return Err(Error::MissingData),
                Some(_) => {
                    // Todo: Improve API for loading input values.
                    let tensor = self
                        .execution_state
                        .get_mut(0)
                        .expect("Provider is hardcoded")
                        .get_value_from_node_id_mut(output)
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
        self.load_input(input)?;

        let provider = self
            .session_state
            ._plan()
            .device(0)
            .expect("We always have one device");

        // Remove allocation.
        for (i, node) in self.session_state.graph().nodes_slice().iter().enumerate() {
            // Todo: We might want to separate the load operation because at this point we don't know the type.
            if matches!(node.op(), Op::NoOp) || matches!(node.op(), Op::Const) {
                continue;
            }
            let kernel = provider.get_kernel(node.op(), DataType::Float)?;

            let mut ctx = Context::new(
                self.execution_state.get_mut(0).ok_or(Error::MissingData)?,
                i,
            )?;

            trace!(
                "{i} {:?} {:?} inputs={:?}",
                node.op(),
                node.def().node.as_ref().unwrap().name,
                node.inputs()
            );

            kernel.compute(&mut ctx)?;
        }

        self.load_output()
    }
}
