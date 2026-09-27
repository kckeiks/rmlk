use crate::core::backend::OperationBackend;
use crate::core::context::Context;
use crate::core::device_service::DeviceService;
use crate::core::error::{BuilderError, InferenceError};
use crate::core::execution_state::ExecutionState;
use crate::core::instance_state::ModelInstanceState;
use crate::core::plan::Plan;
use crate::core::value::Value;
use crate::providers::cuda::Cuda;
use anyhow::anyhow;
use cudarc::driver::CudaContext;
use rmlk_graph::Graph;
use rmlk_schema::{Definition, Op, Tensor};
use std::collections::HashMap;
use std::result;
use std::sync::Arc;

type Result<T> = result::Result<T, InferenceError>;

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
        if input.len() != self.instance_state.graph().input_count() {
            return Err(InferenceError(anyhow!("invalid number of inputs")));
        }

        for (input_name, value) in input {
            let node_id = match self.instance_state.get_input_node_id(&input_name) {
                Some(node_id) => node_id,
                None => {
                    return Err(InferenceError(anyhow!("unknown input: {}", input_name)));
                }
            };

            if self.instance_state.graph().get_node(node_id).is_none() {
                return Err(InferenceError(anyhow!(
                    "failed to find node for `{input_name}` with ID `{node_id}`"
                )));
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
                    return Err(InferenceError(anyhow!(
                        "failed to find output node given ID `{output}`"
                    )));
                }
                Some(node) => {
                    let value = self.execution_state.get_value(output)?;

                    result.insert(
                        node.value()
                            .name()
                            .expect("output nodes should always have a name")
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
        self.execution_state.clear();
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

            if matches!(op, Op::NoOp | Op::Const) {
                continue;
            }

            let mut ctx = Context::new(&mut self.execution_state, id)?;

            if let Err(e) = provider.get_backend(op).and_then(|b| b.compute(&mut ctx)) {
                return Err(InferenceError(e));
            }
        }

        let output = self.get_outputs()?;

        self.clean_up();

        Ok(output)
    }
}

type BuilderResult<T> = result::Result<T, BuilderError>;

/// Builds an instance of a model for inference.
pub struct Builder {
    map_input_name_to_id: HashMap<String, usize>,
    initializers: HashMap<usize, Tensor>,
    graph: Graph<Definition>,
}

impl Builder {
    /// Build from an in-memory [`rmlk_schema::Graph`].
    ///
    /// This is the single entry point into the runtime. Callers that load a
    /// serialized model should deserialize first and then call this.
    pub fn from_graph(graph_schema: rmlk_schema::Graph) -> BuilderResult<Self> {
        let mut map_input_name_to_id = HashMap::new();

        for input in graph_schema.input.as_slice() {
            let name = graph_schema
                .node
                .get(*input)
                .ok_or_else(|| BuilderError::InputNodeNotFound { id: *input })?
                .name
                .as_ref()
                .ok_or_else(|| BuilderError::MissingNodeName { id: *input })?;
            map_input_name_to_id.insert(name.clone(), *input);
        }

        let mut nodes = Vec::with_capacity(graph_schema.node.len());
        for node_schema in graph_schema.node {
            let mut def = Definition::new(node_schema);
            let node = rmlk_graph::Node::new(
                def.take_inputs().unwrap_or_default(),
                def.take_outputs().unwrap_or_default(),
                def,
            );
            nodes.push(node);
        }

        Ok(Self {
            map_input_name_to_id,
            initializers: graph_schema.initializer,
            graph: Graph::new(graph_schema.input, nodes, graph_schema.output),
        })
    }

    pub fn with_model_from_memory(serialized_graph: Box<[u8]>) -> BuilderResult<Self> {
        let graph_schema: rmlk_schema::Graph = bincode::deserialize(serialized_graph.as_ref())
            .map_err(|_| BuilderError::ModelDeserializationFailed)?;
        Self::from_graph(graph_schema)
    }

    pub fn build(self) -> BuilderResult<ModelInstance<Cuda>> {
        let ctx = CudaContext::new(0)
            .map_err(|e| BuilderError::UnexpectedDeviceFailure { error: e.into() })?;

        // Todo: for now everything runs on the default stream.
        let default_stream = ctx.default_stream();

        let provider = Cuda::new(default_stream)?;
        let mut store = provider.store()?;
        store.init(&self.graph, self.initializers)?;

        let plan = Plan::new(Box::new([provider]));
        #[allow(clippy::arc_with_non_send_sync)] // Cuda provider is not Sync yet; ModelInstance should become Send later.
        let instance_state = Arc::new(ModelInstanceState::new(
            plan,
            self.graph,
            self.map_input_name_to_id,
        ));

        Ok(ModelInstance {
            execution_state: ExecutionState::new(instance_state.clone(), store)?,
            instance_state,
        })
    }
}
