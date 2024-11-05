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
use rmlk_schema::{DataType, Definition, Op, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Model instance builder.
pub struct Builder {
    map_io_name_to_id: HashMap<String, usize>,
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
                .ok_or(Error::MissingNode)?
                .name
                .as_ref()
                .ok_or(Error::MissingData)?;
            map_name_to_id.insert(name.clone(), *input);
        }

        for output in graph_schema.output.as_slice() {
            let name = graph_schema
                .node
                .get(*output)
                .ok_or(Error::MissingNode)?
                .name
                .as_ref()
                .ok_or(Error::MissingData)?;
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

    pub fn build(self) -> Result<ModelInstance<Cuda>> {
        let provider = Cuda::new(CudaDevice::new(0)?);
        let values = Values::new(&provider, &self.graph, self.initializers)?;
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
    fn load_input(&mut self, input: HashMap<String, Vec<f32>>) -> Result<()> {
        if input.len() != self.instance_state.graph().inputs().count() {
            return Err(Error::MissingData);
        }

        for (input_name, input_data) in input {
            let provider = self
                .instance_state
                ._plan()
                .device(0)
                .expect("We always have one device");

            let node_id = self
                .instance_state
                .get_io_node_id(&input_name)
                .ok_or(Error::MissingNode)?;

            if self.instance_state.graph().get_node(node_id).is_none() {
                return Err(Error::MissingNode);
            }

            let tensor = self
                .execution_state
                .get_value_from_node_id_mut(node_id)
                .ok_or(Error::MissingData)?;
            tensor.init(provider.htod_float(input_data)?);
        }

        Ok(())
    }

    fn load_output(&mut self) -> Result<Vec<Vec<f32>>> {
        let provider = self
            .instance_state
            ._plan()
            .device(0)
            .expect("We always have one device");

        // Todo: preallocate these buffers.
        let mut result = Vec::with_capacity(self.instance_state.graph().outputs().count());
        for output in self.instance_state.graph().outputs() {
            if self.instance_state.graph().get_node(output).is_none() {
                return Err(Error::MissingData);
            }

            let tensor = self
                .execution_state
                .get_value_from_node_id_mut(output)
                .ok_or(Error::MissingData)?;
            let ptr = tensor.data_mut().take().ok_or(Error::MissingData)?;
            let data = provider.dtoh_float(ptr)?;
            result.push(data);
        }

        Ok(result)
    }

    pub fn run(&mut self, input: HashMap<String, Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        self.load_input(input)?;

        let provider = self
            .instance_state
            ._plan()
            .device(0)
            .expect("We always have one device");

        // Go through the nodes and execute the computation.
        // Todo: use the Plan to know which nodes to compute the output.
        for (id, node) in self.instance_state.graph().node_iter() {
            let op = node.value().op();

            if matches!(op, Op::NoOp) || matches!(op, Op::Const) {
                continue;
            }

            let mut ctx = Context::new(&mut self.execution_state, id)?;

            trace!(
                "{id} {op:?} {:?} inputs={:?}",
                node.value().name(),
                node.inputs()
            );

            provider
                .get_kernel(op, DataType::Float)?
                .compute(&mut ctx)?;
        }

        self.load_output()
    }
}
