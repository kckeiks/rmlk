use crate::common::schema::{attribute_type, Data, GraphDef, NodeTypeInfo, ValueDef};
use rmlk_runtime::Builder;
use rmlk_schema::{DataType, GraphBuilder, Op, Tensor};
use std::collections::{HashMap, HashSet};

mod schema;

/// Build a runtime [`Builder`] from the legacy JSON test dialect.
///
/// Translation goes through [`GraphBuilder`] and [`Builder::from_graph`] so the
/// tests exercise the production entry point. The JSON dialect itself still
/// lives here until the per-op unit-test migration removes it.
pub fn build(test_def: &str) -> Builder {
    let def: GraphDef = serde_json::from_str(test_def).expect("invalid test graph JSON");
    let graph = schema_graph_from_def(def).expect("failed to build schema graph from test JSON");
    Builder::from_graph(graph).expect("Builder::from_graph failed")
}

fn schema_graph_from_def(
    def: GraphDef,
) -> Result<rmlk_schema::Graph, rmlk_schema::GraphBuildError> {
    let GraphDef {
        inputs: named_inputs,
        outputs: named_outputs,
        nodes,
        tensors,
    } = def;

    let input_names: HashSet<String> = named_inputs.iter().cloned().collect();
    let mut tensor_data: HashMap<String, Data> =
        tensors.into_iter().map(|t| (t.name, t.content)).collect();

    let mut g = GraphBuilder::new();

    for node in &nodes {
        if let NodeTypeInfo::Value(ValueDef {
            name,
            shape,
            dtype,
            constant,
        }) = &node.info
        {
            let dtype = dtype.unwrap_or(DataType::Float);
            if input_names.contains(name) {
                g.input(name, dtype, shape.as_deref().unwrap_or(&[]))?;
            } else if constant.unwrap_or(false) || tensor_data.contains_key(name) {
                let content = tensor_data
                    .remove(name)
                    .unwrap_or_else(|| panic!("constant `{name}` has no tensor payload"));
                let dims = shape.clone().unwrap_or_default();
                g.constant(name, tensor_from_data(dims, content))?;
            } else if let Some(shape) = shape {
                g.value(name, dtype, shape)?;
            } else {
                g.value_inferred(name)?;
            }
        }
    }

    for node in nodes {
        if let NodeTypeInfo::Op { name, attributes } = node.info {
            let op: Op = name
                .parse()
                .unwrap_or_else(|_| panic!("unknown op `{name}`"));
            let input_names = node.input.unwrap_or_default();
            let output_names = node.output.unwrap_or_default();
            let inputs: Vec<&str> = input_names.iter().map(String::as_str).collect();
            let outputs: Vec<&str> = output_names.iter().map(String::as_str).collect();

            let mut op_builder = g.op(op, &inputs, &outputs)?;
            if let Some(attrs) = attributes {
                for (attr_name, attr_value) in attrs {
                    op_builder = op_builder.attr(attr_name, attribute_type(attr_value));
                }
            }
        }
    }

    for name in named_outputs {
        g.output(name)?;
    }

    g.build()
}

fn tensor_from_data(dims: Vec<usize>, data: Data) -> Tensor {
    match data {
        Data::Float(values) => Tensor::from_vec(dims, values).expect("float tensor"),
        Data::Double(values) => Tensor::from_vec(dims, values).expect("double tensor"),
        Data::Bool(values) => Tensor::from_vec(dims, values).expect("bool tensor"),
    }
}
