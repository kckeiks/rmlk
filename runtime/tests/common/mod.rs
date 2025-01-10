use crate::common::schema::{GraphDef, NodeTypeInfo};
use rmlk_graph::{Graph, Node};
use rmlk_runtime::Builder;
use rmlk_schema::{DataType, Definition, Op, TypeValue};
use std::collections::HashMap;

mod schema;

pub fn build(test_def: &str) -> Builder {
    let GraphDef {
        inputs: named_inputs,
        outputs: named_outputs,
        nodes: node_defs,
        ..
    } = serde_json::from_str(test_def).unwrap();

    let mut map_io_name_to_id = HashMap::new();
    let mut nodes = Vec::new();
    for node in node_defs {
        let id = nodes.len();
        let mut schema_node = rmlk_schema::Node::new(id);

        match node.info {
            NodeTypeInfo::Op { name } => {
                schema_node.name = Some(name);
                schema_node.op_type = Op::Add;
            }
            NodeTypeInfo::Value { name, shape } => {
                schema_node.name = Some(name);
                if let Some(shape) = shape {
                    schema_node.set_type_value(TypeValue::Tensor {
                        dims: shape,
                        ty: DataType::Float as i32,
                    })
                }
            }
        }

        map_io_name_to_id.insert(schema_node.name.clone().unwrap(), id);

        let mut schema_inputs = Vec::new();
        if let Some(inputs) = node.input {
            for input in inputs {
                let input_id = map_io_name_to_id.get(&input).unwrap();
                schema_inputs.push(*input_id);
            }
            // We don't give ownership of inputs to the schema node because
            // this is what we do in core. Todo: We should probably use a smart
            // pointer to share it cheaply.
        }

        let mut schema_outputs = Vec::new();
        if let Some(outputs) = node.output {
            for output in outputs {
                let output_id = map_io_name_to_id.get(&output).unwrap();
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
        let input_id = map_io_name_to_id.get(&input).unwrap();
        inputs.push(*input_id);
    }
    let mut outputs = Vec::new();
    for output in named_outputs {
        let input_id = map_io_name_to_id.get(&output).unwrap();
        outputs.push(*input_id);
    }

    let graph = Graph::new(inputs, nodes, outputs);

    Builder::new(map_io_name_to_id, HashMap::new(), graph)
}
