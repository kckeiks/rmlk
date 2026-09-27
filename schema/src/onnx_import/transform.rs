use super::traverse::{self, OnnxGraphTraverser, TraversalError};
use crate::onnx::{ModelProto, NodeProto, TensorProto, ValueInfoProto};
use crate::{
    tensor_from_onnx_tensor, Attribute, DataType, Graph, GraphBuildError, GraphBuilder, Op,
    TypeValue,
};
use log::debug;
use quick_protobuf::{BytesReader, MessageRead};
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug)]
pub enum OnnxImportError {
    MissingGraph,
    Parse(String),
    Traversal(TraversalError),
    Build(GraphBuildError),
    Output { name: String, message: String },
}

impl std::fmt::Display for OnnxImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxImportError::MissingGraph => write!(f, "the model does not have a graph"),
            OnnxImportError::Parse(msg) => write!(f, "failed to parse ONNX model: {msg}"),
            OnnxImportError::Traversal(e) => write!(f, "error while traversing the onnx graph: {e}"),
            OnnxImportError::Build(e) => write!(f, "GraphBuilder::build failed: {e}"),
            OnnxImportError::Output { name, message } => {
                write!(f, "failed to register graph output `{name}`: {message}")
            }
        }
    }
}

impl std::error::Error for OnnxImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxImportError::Traversal(e) => Some(e),
            OnnxImportError::Build(e) => Some(e),
            _ => None,
        }
    }
}

struct ModelFromOnnx {
    builder: GraphBuilder,
    /// Names already registered with the builder (inputs, constants, values).
    declared: HashSet<String>,
    /// Graph output names, registered after all ops are wired.
    graph_outputs: Vec<String>,
    base_url: Option<PathBuf>,
}

impl ModelFromOnnx {
    fn new(base_url: Option<PathBuf>) -> Self {
        Self {
            builder: GraphBuilder::new(),
            declared: HashSet::new(),
            graph_outputs: Vec::new(),
            base_url,
        }
    }

    fn ensure_value_inferred(&mut self, name: &str) -> traverse::Result<()> {
        if self.declared.contains(name) {
            return Ok(());
        }
        self.builder.value_inferred(name).map_err(map_build_err)?;
        self.declared.insert(name.to_string());
        Ok(())
    }
}

fn map_build_err(e: GraphBuildError) -> TraversalError {
    TraversalError::InvalidValue(format!("{e:?}"))
}

fn type_value_to_dtype_dims(tv: TypeValue) -> traverse::Result<(DataType, Vec<usize>)> {
    let dtype = DataType::try_from(tv.ty()).map_err(|e| {
        TraversalError::InvalidValue(format!("unsupported dtype {}: {e:?}", tv.ty()))
    })?;
    Ok((dtype, tv.dims().clone()))
}

impl<'a> OnnxGraphTraverser<'a> for ModelFromOnnx {
    fn check_input(&mut self, value_info_proto: ValueInfoProto<'a>) -> traverse::Result<bool> {
        let name = value_info_proto
            .name
            .ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed inputs are not supported".to_string())
            })?
            .into_owned();

        // Initializers are visited first. An ONNX graph input that shares a name
        // with an initializer is a defaulted input — keep the constant only.
        if self.declared.contains(&name) {
            debug!("Skipping graph input `{name}` already registered as initializer");
            return Ok(false);
        }

        debug!("Registering input `{name}`");

        let type_value = TypeValue::from_type_proto(
            value_info_proto
                .type_pb
                .ok_or_else(|| TraversalError::InvalidValue("input missing type".into()))?,
        )
        .map_err(|e| TraversalError::InvalidValue(format!("{e:?}")))?
        .ok_or_else(|| TraversalError::InvalidValue("input type is empty".into()))?;

        let (dtype, dims) = type_value_to_dtype_dims(type_value)?;
        self.builder
            .input(&name, dtype, &dims)
            .map_err(map_build_err)?;
        self.declared.insert(name);
        Ok(false)
    }

    fn check_output(&mut self, value_info_proto: ValueInfoProto<'a>) -> traverse::Result<bool> {
        let name = value_info_proto
            .name
            .ok_or_else(|| {
                TraversalError::InvalidValue("Unnamed outputs are not supported".to_string())
            })?
            .into_owned();

        debug!("Registering output value `{name}`");

        if !self.declared.contains(&name) {
            let type_value = TypeValue::from_type_proto(
                value_info_proto
                    .type_pb
                    .ok_or_else(|| TraversalError::InvalidValue("output missing type".into()))?,
            )
            .map_err(|e| TraversalError::InvalidValue(format!("{e:?}")))?
            .ok_or_else(|| TraversalError::InvalidValue("output type is empty".into()))?;

            let (dtype, dims) = type_value_to_dtype_dims(type_value)?;
            self.builder
                .value(&name, dtype, &dims)
                .map_err(map_build_err)?;
            self.declared.insert(name.clone());
        }

        self.graph_outputs.push(name);
        Ok(false)
    }

    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> traverse::Result<bool> {
        let base_url = self.base_url.clone();
        let name = initializer
            .name
            .clone()
            .ok_or_else(|| {
                TraversalError::InvalidValue("unnamed tensors are not supported".to_string())
            })?
            .into_owned();

        debug!("Registering initializer `{name}`");

        let tensor = tensor_from_onnx_tensor(initializer, base_url)
            .map_err(|e| TraversalError::InvalidValue(format!("{e:?}")))?;

        self.builder
            .constant(&name, tensor)
            .map_err(map_build_err)?;
        if !self.declared.insert(name.clone()) {
            return Err(TraversalError::InvalidValue(format!(
                "duplicate initializer `{name}`"
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

        let input_names: Vec<String> = node_proto.input.iter().map(|s| s.to_string()).collect();
        let output_names: Vec<String> = node_proto.output.iter().map(|s| s.to_string()).collect();

        for name in &input_names {
            if !self.declared.contains(name) {
                return Err(TraversalError::InvalidInnerNode);
            }
        }
        for name in &output_names {
            self.ensure_value_inferred(name)?;
        }

        let input_refs: Vec<&str> = input_names.iter().map(String::as_str).collect();
        let output_refs: Vec<&str> = output_names.iter().map(String::as_str).collect();

        let mut op_builder = self
            .builder
            .op(op, &input_refs, &output_refs)
            .map_err(map_build_err)?;

        for attr_proto in node_proto.attribute {
            let attr = Attribute::try_from(attr_proto)
                .map_err(|e| TraversalError::InvalidValue(format!("{e:?}")))?;
            op_builder = op_builder.attr(attr.name, attr.ty);
        }

        Ok(false)
    }
}

/// Convert a parsed [`ModelProto`] into a [`Graph`] via [`GraphBuilder`].
pub fn graph_from_onnx_proto(
    mut value: ModelProto,
    base_url: Option<PathBuf>,
) -> Result<Graph, OnnxImportError> {
    let mut traverser = ModelFromOnnx::new(base_url);
    let graph_name = value
        .graph
        .as_mut()
        .and_then(|g| g.name.take())
        .map(|name| name.into_owned());
    traverse::visit_onnx(
        value.graph.ok_or(OnnxImportError::MissingGraph)?,
        &mut traverser,
    )
    .map_err(OnnxImportError::Traversal)?;

    for name in &traverser.graph_outputs {
        traverser.builder.output(name).map_err(|e| OnnxImportError::Output {
            name: name.clone(),
            message: e.to_string(),
        })?;
    }

    let mut graph = traverser
        .builder
        .build()
        .map_err(OnnxImportError::Build)?;
    graph.name = graph_name;
    Ok(graph)
}

/// Parse ONNX model bytes and convert them to a [`Graph`].
pub fn graph_from_onnx_bytes(
    data: &[u8],
    base_url: Option<PathBuf>,
) -> Result<Graph, OnnxImportError> {
    let mut reader = BytesReader::from_bytes(data);
    let model_proto = ModelProto::from_reader(&mut reader, data)
        .map_err(|e| OnnxImportError::Parse(e.to_string()))?;
    graph_from_onnx_proto(model_proto, base_url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{
        tensor_proto, tensor_shape_proto, ty_proto, GraphProto, NodeProto, TensorShapeProto,
        TypeProto, ValueInfoProto,
    };
    use crate::{GraphBuilder, Op};
    use std::borrow::Cow;

    fn float_tensor_type(dims: &[i64]) -> TypeProto<'static> {
        TypeProto {
            value: ty_proto::OneOfvalue::tensor_type(ty_proto::Tensor {
                elem_type: Some(tensor_proto::DataType::FLOAT as i32),
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|d| tensor_shape_proto::Dimension {
                            value: tensor_shape_proto::mod_Dimension::OneOfvalue::dim_value(*d),
                            denotation: None,
                        })
                        .collect(),
                }),
            }),
            denotation: None,
        }
    }

    fn value_info(name: &'static str, dims: &[i64]) -> ValueInfoProto<'static> {
        ValueInfoProto {
            name: Some(Cow::Borrowed(name)),
            type_pb: Some(float_tensor_type(dims)),
            doc_string: None,
            metadata_props: vec![],
        }
    }

    /// Import a two-op Add-then-Relu ONNX model and check that the resulting
    /// topology matches a graph built with GraphBuilder.
    #[test]
    fn importer_matches_graph_builder_oracle() {
        let model = ModelProto {
            graph: Some(GraphProto {
                name: Some(Cow::Borrowed("add_relu")),
                input: vec![value_info("a", &[2, 2]), value_info("b", &[2, 2])],
                output: vec![value_info("y", &[2, 2])],
                node: vec![
                    NodeProto {
                        name: Some(Cow::Borrowed("add0")),
                        op_type: Some(Cow::Borrowed("Add")),
                        input: vec![Cow::Borrowed("a"), Cow::Borrowed("b")],
                        output: vec![Cow::Borrowed("t")],
                        ..Default::default()
                    },
                    NodeProto {
                        name: Some(Cow::Borrowed("relu0")),
                        op_type: Some(Cow::Borrowed("Relu")),
                        input: vec![Cow::Borrowed("t")],
                        output: vec![Cow::Borrowed("y")],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };

        let imported = graph_from_onnx_proto(model, None).expect("import");

        let mut g = GraphBuilder::new();
        g.input("a", DataType::Float, [2, 2]).unwrap();
        g.input("b", DataType::Float, [2, 2]).unwrap();
        g.value_inferred("t").unwrap();
        g.value("y", DataType::Float, [2, 2]).unwrap();
        g.op(Op::Add, &["a", "b"], &["t"]).unwrap();
        g.op(Op::Relu, &["t"], &["y"]).unwrap();
        g.output("y").unwrap();
        let mut oracle = g.build().unwrap();
        oracle.name = Some("add_relu".into());

        assert_eq!(imported.input.len(), oracle.input.len());
        assert_eq!(imported.output.len(), oracle.output.len());
        assert_eq!(imported.node.len(), oracle.node.len());
        assert_eq!(imported.name, oracle.name);

        let import_ops: Vec<_> = imported
            .node
            .iter()
            .filter(|n| n.op_type != Op::NoOp)
            .collect();
        let oracle_ops: Vec<_> = oracle
            .node
            .iter()
            .filter(|n| n.op_type != Op::NoOp)
            .collect();
        assert_eq!(import_ops.len(), 2);
        assert_eq!(oracle_ops.len(), 2);
        assert_eq!(import_ops[0].op_type, Op::Add);
        assert_eq!(import_ops[1].op_type, Op::Relu);
        assert_eq!(
            import_ops[0].input.as_ref().map(Vec::len),
            oracle_ops[0].input.as_ref().map(Vec::len)
        );
        assert_eq!(
            import_ops[1].input.as_ref().map(Vec::len),
            oracle_ops[1].input.as_ref().map(Vec::len)
        );
        assert_eq!(import_ops[0].output.as_ref().map(Vec::len), Some(1));
        assert_eq!(import_ops[1].output.as_ref().map(Vec::len), Some(1));
    }
}
