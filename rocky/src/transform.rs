use crate::traverse;
use crate::traverse::{OnnxGraphTraverser, TraversalError};
use anyhow::anyhow;
use log::debug;
use rmlk_schema::onnx::{ModelProto, NodeProto, TensorProto, ValueInfoProto};
use rmlk_schema::{Attribute, Node, Op, Tensor, TypeValue};
use std::collections::HashMap;

#[derive(Default)]
struct ModelFromOnnx {
    nodes: Vec<Node>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    initializers: HashMap<usize, Tensor>,
    name_to_id: HashMap<String, usize>,
}

impl ModelFromOnnx {
    pub fn create_and_get_node(&mut self) -> &mut Node {
        let id = self.nodes.len();
        let node = Node::new(id);
        self.nodes.push(node);
        self.nodes.get_mut(id).expect("We just inserted the node")
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

impl<'a> OnnxGraphTraverser<'a> for ModelFromOnnx {
    fn check_input(&mut self, value_info_proto: ValueInfoProto<'a>) -> traverse::Result<bool> {
        let node = self.create_and_get_node();
        let node_id = node.id;

        debug!("Assigning id={node_id} for input {value_info_proto:?}");

        let name = {
            let name = value_info_proto.name.ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed inputs are not supported".to_string())
            })?;
            name
        };
        // Todo: avoid allocation.
        node.set_name(name.to_string());
        node.set_op(Op::NoOp);
        // Todo: Handle unwrap().
        let type_value = TypeValue::from_type_proto(value_info_proto.type_pb.unwrap())
            .unwrap()
            .unwrap();
        node.set_type_value(type_value);

        self.add_input(node_id);

        if let Some(old_id) = self.save_name_to_id(name.into_owned(), node_id) {
            return Err(TraversalError::InvalidValue(format!(
                "found two inputs with the same for id: prev:[{old_id}] new:[{}]",
                node_id
            )));
        }

        Ok(false)
    }

    fn check_output(&mut self, value_info_proto: ValueInfoProto<'a>) -> traverse::Result<bool> {
        let node = self.create_and_get_node();
        let node_id = node.id;

        debug!("Assigning id={node_id} for output {value_info_proto:?}");

        let name = {
            let name = value_info_proto.name.ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed outputs are not supported".to_string())
            })?;
            name
        };
        // Todo: avoid allocation.
        node.set_name(name.to_string());
        node.set_op(Op::NoOp);
        // Todo: Handle unwrap().
        let type_value = TypeValue::from_type_proto(value_info_proto.type_pb.unwrap())
            .unwrap()
            .unwrap();
        node.set_type_value(type_value);

        self.add_output(node_id);

        if let Some(old_id) = self.save_name_to_id(name.into_owned(), node_id) {
            return Err(TraversalError::InvalidValue(format!(
                "found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]"
            )));
        }

        Ok(false)
    }

    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> traverse::Result<bool> {
        let node = self.create_and_get_node();
        let node_id = node.id;

        debug!(
            "Assigning id={node_id} for initializer {:?}",
            initializer.name
        );

        let name = initializer.name.clone().ok_or_else(|| {
            TraversalError::InvalidValue("unnamed tensors are not supported".to_string())
        })?;
        // Todo: avoid allocation.
        node.set_name(name.to_string());
        node.set_op(Op::NoOp);

        let tensor = Tensor::from_onnx_tensor(initializer).unwrap();

        let type_value = TypeValue::Tensor {
            ty: tensor.data_type as i32,
            dims: tensor.dims.clone(),
        };
        node.set_type_value(type_value);

        self.add_initializer(node_id, tensor);

        if let Some(old_id) = self.save_name_to_id(name.into_owned(), node_id) {
            return Err(TraversalError::InvalidValue(format!(
                "found two initializers with the same for id: prev:[{old_id}] new:[{node_id}]"
            )));
        }

        Ok(false)
    }

    fn check_inner_node(&mut self, node_proto: NodeProto<'a>) -> traverse::Result<bool> {
        let op = node_proto
            .op_type
            .as_ref()
            .map(|op| op.parse::<Op>())
            .ok_or(TraversalError::InvalidInnerNode)?
            .map_err(|_| TraversalError::InvalidInnerNode)?;

        let node = self.create_and_get_node();
        node.set_op(op);
        // Todo: remove allocation.
        node.set_name(node_proto.name.unwrap().to_string());

        let mut attributes = Vec::new();
        for attr_proto in node_proto.attribute {
            attributes.push(Attribute::try_from(attr_proto).unwrap());
        }

        node.set_attributes(attributes);

        let node_id = node.id;
        let mut inputs = Vec::new();
        for name in node_proto.input {
            // Todo: Mapping one name to a single node id, we lose information,
            // because a single node might have two outputs, how do we differentiate?
            let input_node_id = self
                .get_node_id(name.as_ref())
                .ok_or_else(|| TraversalError::InvalidInnerNode)?;
            inputs.push(input_node_id);
        }
        let node = self.get_node_mut(node_id).expect("We just inserted it");
        node.set_inputs(inputs);

        let mut output_node_ids = Vec::new();
        for name in node_proto.output {
            match self.get_node_id(&name) {
                None => {
                    let output_node = self.create_and_get_node();
                    output_node.add_input(node_id);
                    output_node.set_name(name.to_string());

                    output_node_ids.push(output_node.id);

                    let output_node_id = output_node.id;
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

pub fn graph_from_onnx_proto(mut value: ModelProto) -> anyhow::Result<rmlk_schema::Graph> {
    let mut traverser = ModelFromOnnx::default();
    let graph_name = value
        .graph
        .as_mut()
        .map(|g| g.name.take())
        .flatten()
        .map(|name| name.into_owned());
    traverse::visit_onnx(
        value
            .graph
            .ok_or(anyhow!("the model does not have a graph"))?,
        &mut traverser,
    )
    .map_err(|e| anyhow!("an error ocurred while traversing the onnx graph: {e:?}"))?;
    Ok(rmlk_schema::Graph {
        node: traverser.nodes,
        name: graph_name,
        initializer: traverser.initializers,
        input: traverser.inputs,
        output: traverser.outputs,
        quantization_annotation: None,
    })
}
