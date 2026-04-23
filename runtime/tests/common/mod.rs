use crate::common::schema::{parse_attributes, GraphDef, NodeTypeInfo, ValueDef};
use rmlk_graph::{Graph, Node};
use rmlk_runtime::Builder;
use rmlk_schema::{DataType, Definition, Op, Tensor, TypeValue};
use std::collections::HashMap;

mod schema;

pub fn build(test_def: &str) -> Builder {
    let GraphDef {
        inputs: named_inputs,
        outputs: named_outputs,
        nodes: node_defs,
        tensors,
    } = serde_json::from_str(test_def).unwrap();

    let mut map_name_to_id = HashMap::new();
    let mut map_input_name_to_id = HashMap::new();
    let mut nodes = Vec::new();
    for node in node_defs {
        let id = nodes.len();
        let mut schema_node = rmlk_schema::Node::new(id);

        match node.info {
            NodeTypeInfo::Op { name, attributes } => {
                schema_node.op_type = name.parse().unwrap();
                schema_node.name = Some(name);

                if let Some(attrs) = attributes {
                    schema_node.attribute = Some(parse_attributes(attrs));
                }
            }
            NodeTypeInfo::Value(ValueDef {
                name,
                shape,
                dtype,
                constant,
            }) => {
                schema_node.name = Some(name);

                if constant.unwrap_or(false) {
                    schema_node.op_type = Op::NoOp;
                }

                // Todo: circle back and assess this code.
                if let Some(shape) = shape {
                    schema_node.set_type_value(TypeValue::Tensor {
                        dims: shape,
                        ty: dtype.unwrap_or(DataType::Float).into(),
                        has_dynamic_dims: false,
                    })
                }
            }
        }

        map_name_to_id.insert(schema_node.name.clone().unwrap(), id);

        let mut schema_inputs = Vec::new();
        if let Some(inputs) = node.input {
            for input in inputs {
                let input_id = map_name_to_id.get(&input).unwrap();
                schema_inputs.push(*input_id);
            }
            // We don't give ownership of inputs to the schema node because
            // this is what we do in core. Todo: We should probably use a smart
            // pointer to share it cheaply.
        }

        let mut schema_outputs = Vec::new();
        if let Some(outputs) = node.output {
            for output in outputs {
                let output_id = map_name_to_id.get(&output).unwrap();
                schema_outputs.push(*output_id);
            }
            // We don't give ownership of inputs to the schema node because
            // this is what we do in core. Todo: We should probably use a smart
            // pointer to share it cheaply.
        }

        let definition = Definition::new(schema_node);

        nodes.push(Node::new(schema_inputs, schema_outputs, definition));
    }

    let mut inputs = Vec::new();
    for input in named_inputs {
        let input_id = map_name_to_id.get(&input).unwrap();
        map_input_name_to_id.insert(input.clone(), *input_id);
        inputs.push(*input_id);
    }

    let mut outputs = Vec::new();
    for output in named_outputs {
        let output_id = map_name_to_id.get(&output).unwrap();
        outputs.push(*output_id);
    }

    let mut initializers = HashMap::new();
    for tensor in tensors {
        let id = map_name_to_id.get(&tensor.name).unwrap();
        let node_info = nodes.get(*id).unwrap();
        let node_def = node_info.value();

        let tensor = Tensor {
            dims: node_def.shape().unwrap().clone(),
            data_type: node_def.dtype().unwrap(),
            segment: None,
            float_data: tensor.content.float(),
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![],
            name: Some(tensor.name.clone()),
            doc_string: None,
            raw_data: None,
            double_data: tensor.content.double(),
            uint64_data: vec![],
            bool_data: tensor.content.bool(),
        };

        initializers.insert(*id, tensor);
    }

    let graph = Graph::new(inputs, nodes, outputs);

    Builder::new(map_input_name_to_id, initializers, graph)
}
