use crate::core::backend::OperationBackend;
use crate::core::context::Context;
use crate::core::device_service::DeviceService;
use crate::core::error::{Error, InternalError};
use crate::core::execution_state::ExecutionState;
use crate::core::instance_state::ModelInstanceState;
use crate::core::plan::Plan;
use crate::core::store::TensorStore;
use crate::core::value::Value;
use crate::providers::cuda::Cuda;
use cudarc::driver::CudaDevice;
use log::{debug, trace};
use rmlk_graph::Graph;
use rmlk_schema::{DataType, Definition, Op, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Error>;

/// Builds an instance of a model for inference.
pub struct Builder {
    /// Input/output names to node ID map.
    map_io_name_to_id: HashMap<String, usize>,
    /// In
    initializers: HashMap<usize, Tensor>,
    graph: Graph<Definition>,
}

impl Builder {
    pub fn with_model_from_memory(serialized_graph: Box<[u8]>) -> Result<Self> {
        let graph_schema: rmlk_schema::Graph = bincode::deserialize(serialized_graph.as_ref())
            .map_err(|_| Error::ModelDeserializationFailed)?;

        let mut map_name_to_id = HashMap::new();

        for input in graph_schema.input.as_slice() {
            let name = graph_schema
                .node
                .get(*input)
                .ok_or_else(|| Error::NodeNotFound { id: *input })?
                .name
                .as_ref()
                .ok_or_else(|| Error::ExpectedName { node_id: *input })?;
            map_name_to_id.insert(name.clone(), *input);
        }

        for output in graph_schema.output.as_slice() {
            let name = graph_schema
                .node
                .get(*output)
                .ok_or_else(|| Error::NodeNotFound { id: *output })?
                .name
                .as_ref()
                .ok_or_else(|| Error::ExpectedName { node_id: *output })?;
            map_name_to_id.insert(name.clone(), *output);
        }

        let mut nodes = Vec::with_capacity(graph_schema.node.len());
        for node_schema in graph_schema.node {
            debug_assert!(node_schema.id == nodes.len());
            let mut def = Definition::new(node_schema);
            let node = rmlk_graph::Node::new(
                def.take_inputs().unwrap_or_default(),
                def.take_outputs().unwrap_or_default(),
                def,
            );
            nodes.push(node);
        }

        Ok(Self {
            map_io_name_to_id: map_name_to_id,
            initializers: graph_schema.initializer,
            graph: Graph::new(graph_schema.input, nodes, graph_schema.output),
        })
    }

    // Todo: we should put this behind a flag for testing only.
    pub fn new(
        map_io_name_to_id: HashMap<String, usize>,
        initializers: HashMap<usize, Tensor>,
        graph: Graph<Definition>,
    ) -> Self {
        Self {
            map_io_name_to_id,
            initializers,
            graph,
        }
    }

    pub fn build(self) -> Result<ModelInstance<Cuda>> {
        let provider = Cuda::new(
            CudaDevice::new(0)
                .map_err(rmlk_cuda::Error::from)
                .map_err(InternalError::from)?,
        );
        let values = TensorStore::new(&provider, &self.graph, self.initializers)?;
        let plan = Plan::new(Box::new([provider]));
        let instance_state = Arc::new(ModelInstanceState::new(
            plan,
            self.graph,
            self.map_io_name_to_id,
        ));

        Ok(ModelInstance {
            execution_state: ExecutionState::new(instance_state.clone(), values)?,
            instance_state,
        })
    }
}

/// A model instance.
///
/// This object represents the model instantiated in the
/// runtime.
pub struct ModelInstance<D: DeviceService> {
    instance_state: Arc<ModelInstanceState<D>>,
    execution_state: ExecutionState<D>,
}

impl<D> ModelInstance<D>
where
    D: DeviceService,
{
    fn load_inputs(&mut self, input: HashMap<String, Value>) -> Result<()> {
        if input.len() != self.instance_state.graph().inputs().count() {
            debug!("user input: {input:?}");
            return Err(Error::InvalidUserInput { input });
        }

        for (input_name, value) in input {
            let node_id = self
                .instance_state
                .get_io_node_id(&input_name)
                .ok_or_else(|| Error::FailedToFindNodeId { name: input_name })?;

            if self.instance_state.graph().get_node(node_id).is_none() {
                return Err(Error::NodeNotFound { id: node_id });
            }

            self.execution_state.load_value(node_id, value)?;
        }

        Ok(())
    }

    fn get_outputs(&mut self) -> Result<HashMap<String, Value>> {
        let mut result = HashMap::new();
        for output in self.instance_state.graph().outputs() {
            match self.instance_state.graph().get_node(output) {
                None => {
                    return Err(Error::NodeNotFound { id: output });
                }
                Some(node) => {
                    let value = self.execution_state.get_value(output)?;
                    result.insert(
                        node.value()
                            .name()
                            .ok_or_else(|| Error::ExpectedName { node_id: output })?
                            .to_string(),
                        value,
                    );
                }
            }
        }

        Ok(result)
    }

    fn clean_up(&mut self) {
        self.execution_state.scratch_alloc_mut().reset();
    }

    pub fn run(&mut self, input: HashMap<String, Value>) -> Result<HashMap<String, Value>> {
        self.load_inputs(input)?;

        let provider = self
            .instance_state
            ._plan()
            .device(0)
            .expect("we always have one device");

        // Go through the nodes and execute the computation.
        // Todo: use the Plan to know which nodes to compute the output.
        for (id, node) in self.instance_state.graph().node_iter() {
            let op = node.value().op();

            if matches!(op, Op::NoOp) {
                continue;
            }

            let mut ctx = Context::new(&mut self.execution_state, id)?;

            trace!(
                "[node={id}][{op:?}][name={:?}][inputs={:?}][outputs={:?}]",
                node.value().name(),
                node.inputs(),
                node.outputs(),
            );

            // Todo: we need to spec this out.
            let dtype = match ctx.get_input(0) {
                Ok(tensor) => tensor.dtype(),
                Err(_) => {
                    debug!("[{op:?}] no input found");
                    DataType::Undefined
                }
            };

            provider.get_backend(op, dtype)?.compute(&mut ctx)?;
        }

        let output = self.get_outputs()?;

        self.clean_up();

        Ok(output)
    }
}
