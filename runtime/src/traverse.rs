use log::debug;
use rmlk_graph::{Definition, Graph};
use rmlk_schema::onnx::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use rmlk_schema::{Attribute, Node, Op, Tensor, TypeValue};
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, TraversalError>;

#[derive(Debug)]
pub enum TraversalError {
    #[allow(unused)]
    InvalidValue(String),
    InvalidInnerNode,
}

pub trait OnnxGraphTraverser<'a> {
    fn check_input(&mut self, input: ValueInfoProto<'a>) -> Result<bool>;
    fn check_output(&mut self, output: ValueInfoProto<'a>) -> Result<bool>;
    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> Result<bool>;
    fn check_inner_node(&mut self, node: NodeProto<'a>) -> Result<bool>;
}

pub fn visit_onnx<'a, T>(graph_proto: GraphProto<'a>, traverser: &mut T) -> Result<()>
where
    T: OnnxGraphTraverser<'a>,
{
    for initializer in graph_proto.initializer {
        if traverser.check_initializer(initializer)? {
            return Ok(());
        }
    }

    for input in graph_proto.input {
        if traverser.check_input(input)? {
            return Ok(());
        }
    }

    for output in graph_proto.output {
        if traverser.check_output(output)? {
            return Ok(());
        }
    }

    for node in graph_proto.node {
        if traverser.check_inner_node(node)? {
            return Ok(());
        }
    }

    Ok(())
}

#[derive(Default)]
pub struct GraphFromOnnx {
    pub debug_mode: bool,
    pub nodes: Vec<Node>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub initializers: HashMap<usize, Tensor>,
    pub name_to_id: HashMap<String, usize>,
}

impl GraphFromOnnx {
    pub fn next_id(&self) -> usize {
        self.nodes.len()
    }

    pub fn add_node(&mut self, node: Node) -> usize {
        let id = self.next_id();
        self.nodes.push(node);
        id
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    pub fn add_input(&mut self, input: usize) {
        self.inputs.push(input);
    }

    pub fn add_output(&mut self, output: usize) {
        self.outputs.push(output);
    }

    pub fn add_initializer(&mut self, id: usize, initializers: Tensor) -> Option<Tensor> {
        self.initializers.insert(id, initializers)
    }

    pub fn get_node_id(&self, name: &str) -> Option<usize> {
        self.name_to_id.get(name).copied()
    }

    pub fn save_name_to_id(&mut self, name: String, id: usize) -> Option<usize> {
        self.name_to_id.insert(name, id)
    }
}

impl<'a> OnnxGraphTraverser<'a> for GraphFromOnnx {
    fn check_input(&mut self, value_info_proto: ValueInfoProto<'a>) -> Result<bool> {
        let node_id = self.next_id();

        debug!("Assigning id={node_id} for input {value_info_proto:?}");

        let name = {
            let name = value_info_proto.name.ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed inputs are not supported".to_string())
            })?;
            name
        };

        let mut node = Node::new(node_id);
        node.set_op(Op::NoOp);

        // Todo: Handle unwrap().
        let type_value = TypeValue::from_type_proto(value_info_proto.type_pb.unwrap())
            .unwrap()
            .unwrap();
        node.set_type_value(type_value);

        // Todo: avoid allocation.
        node.set_name(name.to_string());

        let node_id = self.add_node(node);

        self.add_input(node_id);

        if let Some(old_id) = self.save_name_to_id(name.into_owned(), node_id) {
            return Err(TraversalError::InvalidValue(format!(
                "found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]"
            )));
        }

        Ok(false)
    }

    fn check_output(&mut self, value_info_proto: ValueInfoProto<'a>) -> Result<bool> {
        let node_id = self.next_id();

        debug!("Assigning id={node_id} for output {value_info_proto:?}");

        let name = {
            let name = value_info_proto.name.ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed outputs are not supported".to_string())
            })?;
            name
        };

        let mut node = Node::new(node_id);
        node.set_op(Op::NoOp);

        let type_value = TypeValue::from_type_proto(value_info_proto.type_pb.unwrap())
            .unwrap()
            .unwrap();
        node.set_type_value(type_value);

        // Todo: avoid allocation.
        node.set_name(name.to_string());

        let node_id = self.add_node(node);

        self.add_output(node_id);

        if let Some(old_id) = self.save_name_to_id(name.into_owned(), node_id) {
            return Err(TraversalError::InvalidValue(format!(
                "found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]"
            )));
        }

        Ok(false)
    }

    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> Result<bool> {
        let node_id = self.next_id();

        debug!(
            "Assigning id={node_id} for initializer {:?}",
            initializer.name
        );

        let name = initializer.name.clone().ok_or_else(|| {
            TraversalError::InvalidValue("unnamed tensors are not supported".to_string())
        })?;

        let mut node = Node::new(node_id);
        node.set_op(Op::Const);

        let tensor = Tensor::from_onnx_tensor(initializer).unwrap();

        let type_value = TypeValue::Tensor {
            ty: tensor.data_type as i32,
            dims: tensor.dims.clone(),
        };
        node.set_type_value(type_value);

        // Todo: avoid allocation.
        node.set_name(name.to_string());

        let node_id = self.add_node(node);

        self.add_initializer(node_id, tensor);

        if let Some(old_id) = self.save_name_to_id(name.into_owned(), node_id) {
            return Err(TraversalError::InvalidValue(format!(
                "found two initializers with the same for id: prev:[{old_id}] new:[{node_id}]"
            )));
        }

        Ok(false)
    }

    fn check_inner_node(&mut self, node_proto: NodeProto<'a>) -> Result<bool> {
        let op = node_proto
            .op_type
            .as_ref()
            .map(|op| op.parse::<Op>())
            .ok_or(TraversalError::InvalidInnerNode)?
            .map_err(|_| TraversalError::InvalidInnerNode)?;

        let node_id = self.next_id();
        let mut node = Node::new(node_id);
        node.set_op(op);
        // Todo: remove allocation.
        node.set_name(node_proto.name.unwrap().to_string());

        let mut attributes = Vec::new();
        for attr_proto in node_proto.attribute {
            attributes.push(Attribute::try_from(attr_proto).unwrap());
        }

        node.set_attributes(attributes);

        let mut inputs = Vec::new();
        for name in node_proto.input {
            // Todo: Mapping one name to a single node id, we lose information,
            // because a single node might have two outputs, how do we differentiate?
            let input_node_id = self
                .get_node_id(name.as_ref())
                .ok_or_else(|| TraversalError::InvalidInnerNode)?;
            inputs.push(input_node_id);
        }

        node.set_inputs(inputs);

        let node_id = self.add_node(node);

        let mut output_node_ids = Vec::new();
        for name in node_proto.output {
            match self.get_node_id(&name) {
                None => {
                    let id = self.next_id();
                    let mut output_node = Node::new(id);
                    output_node.add_input(node_id);
                    output_node.set_name(name.to_string());

                    let output_node_id = self.add_node(output_node);
                    output_node_ids.push(output_node_id);

                    // Todo: remove clone.
                    self.save_name_to_id(name.into_owned(), output_node_id);
                }
                Some(id) => {
                    let node = self
                        .get_node_mut(id)
                        .ok_or(TraversalError::InvalidInnerNode)?;
                    node.add_input(node_id);
                    output_node_ids.push(id);
                }
            }
        }

        let node = self
            .get_node_mut(node_id)
            .expect("We just inserted it above.");
        for id in output_node_ids {
            node.add_output(id);
        }

        Ok(false)
    }
}

impl From<GraphFromOnnx> for Graph<Definition> {
    fn from(value: GraphFromOnnx) -> Self {
        let mut nodes = Vec::new();
        for node_schema in value.nodes {
            debug_assert!(node_schema.id == nodes.len());
            let mut def = Definition::new(node_schema);
            let node = rmlk_graph::Node::from_definition(
                def.take_inputs().unwrap_or_default(),
                def.take_outputs().unwrap_or_default(),
                def,
            );
            nodes.push(node);
        }

        Self::new(value.initializers, value.inputs, nodes, value.outputs)
    }
}
