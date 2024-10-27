use log::{debug, error};
use rmlk_graph::{Definition, Graph, OnnxGraphTraverser, TraversalError};
use rmlk_ir::ir_v2::{Attribute, Node, TypeValue, Value, ValueInfo};
use rmlk_ir::{NodeProto, Op, Tensor, TensorProto, ValueInfoProto};
use std::borrow::Cow;
use std::collections::HashMap;

pub struct GraphFromOnnxV2 {
    pub debug_mode: bool,
    pub nodes: Vec<Node>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub initializers: HashMap<usize, Tensor>,
    pub name_to_id: HashMap<String, usize>,
}

impl GraphFromOnnxV2 {
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

impl<'a> OnnxGraphTraverser<'a> for GraphFromOnnxV2 {
    fn check_input(
        &mut self,
        value_info_proto: ValueInfoProto<'a>,
    ) -> Result<bool, TraversalError> {
        let node_id = self.next_id();

        debug!("Assigning id={node_id} for input {value_info_proto:?}");

        let name = {
            let name = value_info_proto.name.ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed inputs are not supported".to_string())
            })?;
            debug_assert!(matches!(name, Cow::Owned(_)));
            name
        };

        let mut node = Node::new(node_id);
        node.set_op(Op::NoOp.into());

        // Todo: Handle unwrap().
        let parsed_type = rmlk_ir::Type::try_from(value_info_proto.type_pb.unwrap()).unwrap();
        let (dtype, dims) = parsed_type
            .get_tensor_info()
            .ok_or_else(|| TraversalError::InvalidValue("missing tensor info".to_string()))?;

        let type_value = TypeValue::Tensor {
            ty: dtype as i32,
            dims: dims.unwrap(),
        };
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

    fn check_output(
        &mut self,
        value_info_proto: ValueInfoProto<'a>,
    ) -> Result<bool, TraversalError> {
        let node_id = self.next_id();

        debug!("Assigning id={node_id} for output {value_info_proto:?}");

        let name = {
            let name = value_info_proto.name.ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed outputs are not supported".to_string())
            })?;
            debug_assert!(matches!(name, Cow::Owned(_)));
            name
        };

        let mut node = Node::new(node_id);
        node.set_op(Op::NoOp as u32);

        let parsed_type = rmlk_ir::Type::try_from(value_info_proto.type_pb.unwrap()).unwrap();
        let (dtype, dims) = parsed_type
            .get_tensor_info()
            .ok_or_else(|| TraversalError::InvalidValue("missing tensor info".to_string()))?;

        let type_value = TypeValue::Tensor {
            ty: dtype as i32,
            dims: dims.unwrap(),
        };
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

    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> Result<bool, TraversalError> {
        let node_id = self.next_id();

        debug!(
            "Assigning id={node_id} for initializer {:?}",
            initializer.name
        );

        let name = initializer.name.clone().ok_or_else(|| {
            TraversalError::InvalidValue("unnamed tensors are not supported".to_string())
        })?;

        let mut node = Node::new(node_id);
        node.set_op(Op::Const as u32);

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

    fn check_inner_node(&mut self, node_proto: NodeProto<'a>) -> Result<bool, TraversalError> {
        let op = node_proto
            .op_type
            .as_ref()
            .map(|op| op.parse::<Op>())
            .ok_or(TraversalError::InvalidInnerNode)?
            .map_err(|_| TraversalError::InvalidInnerNode)?;

        let node_id = self.next_id();
        let mut node = Node::new(node_id);
        node.set_op(op as u32);
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

impl From<GraphFromOnnxV2> for Graph {
    fn from(value: GraphFromOnnxV2) -> Self {
        let mut nodes = Vec::new();
        for node_schema in value.nodes {
            debug_assert!(node_schema.id == nodes.len());
            let def = Definition::new(node_schema);
            let node = rmlk_graph::Node::from_definition(def);
            nodes.push(node);
        }

        Self::new(value.initializers, value.inputs, nodes, value.outputs)
    }
}
