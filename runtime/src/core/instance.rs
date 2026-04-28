use crate::core::backend::OperationBackend;
use crate::core::context::Context;
use crate::core::device_service::DeviceService;
use crate::core::error::{BuilderError, Error};
use crate::core::execution_state::ExecutionState;
use crate::core::instance_state::ModelInstanceState;
use crate::core::value::Value;
use anyhow::anyhow;
use rmlk_schema::{Definition, Op, Tensor};
use std::collections::HashMap;
use std::result;
use std::sync::Arc;
use cudarc::driver::CudaContext;
use rmlk_graph::Graph;
use crate::core::plan::Plan;
use crate::providers::cuda::Cuda;

type Result<T> = std::result::Result<T, Error>;

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
            let mut expected = Vec::new();
            for input in self.instance_state.graph().inputs() {
                match self
                    .instance_state
                    .graph()
                    .get_node(input)
                    .and_then(|n| n.value().name())
                {
                    Some(name) => expected.push(name.to_string()),
                    None => {
                        return Err(Error::InferenceError {
                            error: anyhow!("failed to find expected input node with ID `{input}`")
                        })
                    }
                }
            }
            return Err(Error::InvalidInputs {
                received: input,
                expected,
            });
        }

        for (input_name, value) in input {
            let node_id = match self.instance_state.get_input_node_id(&input_name) {
                Some(node_id) => node_id,
                None => return Err(Error::UnknownInputName { name: input_name }),
            };

            if self.instance_state.graph().get_node(node_id).is_none() {
                return Err(Error::InferenceError {
                    error: anyhow!("failed to find node `{input_name}` with ID `{node_id}`")
                });
            }

            self.execution_state
                .load_value(node_id, value)
                .map_err(|e| Error::InferenceError {
                    error: e
                })?;
        }

        Ok(())
    }

    fn get_outputs(&mut self) -> Result<HashMap<String, Value>> {
        let mut result = HashMap::new();
        for output in self.instance_state.graph().outputs() {
            match self.instance_state.graph().get_node(output) {
                None => {
                    return Err(Error::InferenceError {
                        error: anyhow!("failed to find output node given ID `{output}`")
                    });
                }
                Some(node) => {
                    let value =
                        self.execution_state
                            .get_value(output)
                            .map_err(|e| Error::InferenceError {
                                error: e,
                            })?;

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

            let mut ctx =
                Context::new(&mut self.execution_state, id).map_err(|e| Error::InferenceError {
                    error: e,
                })?;

            if let Err(e) = provider.get_backend(op).and_then(|b| b.compute(&mut ctx)) {
                return Err(Error::InferenceError { error: e });
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
    pub fn with_model_from_memory(serialized_graph: Box<[u8]>) -> BuilderResult<Self> {
        let graph_schema: rmlk_schema::Graph = bincode::deserialize(serialized_graph.as_ref())
            .map_err(|_| BuilderError::ModelDeserializationFailed)?;

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

        // Use an allocator here.
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

    // Todo: we should put this behind a flag for testing only.
    pub fn new(
        map_input_name_to_id: HashMap<String, usize>,
        initializers: HashMap<usize, Tensor>,
        graph: Graph<Definition>,
    ) -> Self {
        Self {
            map_input_name_to_id,
            initializers,
            graph,
        }
    }

    pub fn build(self) -> BuilderResult<ModelInstance<Cuda>> {
        let ctx = CudaContext::new(0)
            .map_err(|e| BuilderError::UnexpectedDeviceFailure { error: e.into() })?;

        // Todo: for now everything runs on the default stream.
        let default_stream = ctx.default_stream();

        let provider = Cuda::new(default_stream)?;
        let mut store = provider.store()?;
        store
            .init(&self.graph, self.initializers)?;

        let plan = Plan::new(Box::new([provider]));
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