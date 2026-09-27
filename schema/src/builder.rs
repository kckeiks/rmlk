use crate::attributes::{Attribute, AttributeType};
use crate::graph::Graph;
use crate::node::Node;
use crate::tensor::Tensor;
use crate::value::TypeValue;
use crate::{DataType, Op};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Errors from [`GraphBuilder`].
#[derive(Debug, PartialEq)]
pub enum GraphBuildError {
    DuplicateName(String),
    UnknownName(String),
    /// A graph output that is not an input, constant, or result of any op.
    OutputNotProduced(String),
    /// `build` was called with no `output(...)` registrations.
    NoOutputs,
    /// `op` was called with [`Op::NoOp`].
    InvalidOp(Op),
    /// `op` was called with an empty outputs list.
    OpMissingOutputs,
}

impl fmt::Display for GraphBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "duplicate node name `{name}`"),
            Self::UnknownName(name) => write!(f, "unknown name `{name}`"),
            Self::OutputNotProduced(name) => {
                write!(f, "output `{name}` is not produced by any op")
            }
            Self::NoOutputs => write!(f, "graph has no outputs"),
            Self::InvalidOp(op) => write!(f, "invalid op for GraphBuilder::op: {op:?}"),
            Self::OpMissingOutputs => write!(f, "op must declare at least one output"),
        }
    }
}

impl std::error::Error for GraphBuildError {}

/// Name-based builder that produces an [`Graph`].
///
/// Value nodes (`input`, `constant`, `value`) must be declared before an `op`
/// references them. Edges are stored on the schema nodes the same way the ONNX
/// importer does, so the result is ready for the production runtime entry point.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    name_to_id: HashMap<String, usize>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    initializers: HashMap<usize, Tensor>,
    /// Value nodes available without an op (graph inputs and constants).
    sources: HashSet<usize>,
    /// Value nodes written by at least one op.
    produced: HashSet<usize>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a graph input value.
    pub fn input(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        dims: impl AsRef<[usize]>,
    ) -> Result<&mut Self, GraphBuildError> {
        let id = self.add_value_node(name, dtype, dims.as_ref().to_vec())?;
        self.inputs.push(id);
        self.sources.insert(id);
        Ok(self)
    }

    /// Declare a constant value and attach its initializer tensor.
    ///
    /// Shape and dtype are taken from `tensor`. The tensor's `name` is set to
    /// the builder name.
    pub fn constant(
        &mut self,
        name: impl Into<String>,
        mut tensor: Tensor,
    ) -> Result<&mut Self, GraphBuildError> {
        let name = name.into();
        let id = self.add_value_node(name.clone(), tensor.data_type, tensor.dims.clone())?;
        tensor.name = Some(name);
        self.initializers.insert(id, tensor);
        self.sources.insert(id);
        Ok(self)
    }

    /// Declare an intermediate value that an op will write.
    pub fn value(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        dims: impl AsRef<[usize]>,
    ) -> Result<&mut Self, GraphBuildError> {
        self.add_value_node(name, dtype, dims.as_ref().to_vec())?;
        Ok(self)
    }

    /// Append an operator node.
    ///
    /// `inputs` and `outputs` are value names declared earlier. Returns an
    /// [`OpBuilder`] for attaching attributes.
    pub fn op(
        &mut self,
        op: Op,
        inputs: &[&str],
        outputs: &[&str],
    ) -> Result<OpBuilder<'_>, GraphBuildError> {
        if matches!(op, Op::NoOp) {
            return Err(GraphBuildError::InvalidOp(op));
        }
        if outputs.is_empty() {
            return Err(GraphBuildError::OpMissingOutputs);
        }

        let input_ids = self.resolve_names(inputs)?;
        let output_ids = self.resolve_names(outputs)?;

        let id = self.nodes.len();
        let mut node = Node::new(id);
        node.set_op(op);
        node.set_name(format!("{op:?}_{id}"));
        node.set_inputs(input_ids);
        node.set_outputs(output_ids.clone());
        self.nodes.push(node);

        for output_id in &output_ids {
            self.nodes[*output_id].add_input(id);
            self.produced.insert(*output_id);
        }

        Ok(OpBuilder {
            builder: self,
            op_id: id,
        })
    }

    /// Mark a previously declared value as a graph output.
    pub fn output(&mut self, name: impl Into<String>) -> Result<&mut Self, GraphBuildError> {
        let name = name.into();
        let id = self
            .name_to_id
            .get(&name)
            .copied()
            .ok_or_else(|| GraphBuildError::UnknownName(name))?;
        self.outputs.push(id);
        Ok(self)
    }

    /// Finish the graph.
    ///
    /// Fails if there are no outputs, or if any output is neither a source
    /// (input/constant) nor produced by an op.
    pub fn build(self) -> Result<Graph, GraphBuildError> {
        if self.outputs.is_empty() {
            return Err(GraphBuildError::NoOutputs);
        }

        for &out_id in &self.outputs {
            if !self.produced.contains(&out_id) && !self.sources.contains(&out_id) {
                let name = self.nodes[out_id]
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("#{out_id}"));
                return Err(GraphBuildError::OutputNotProduced(name));
            }
        }

        Ok(Graph {
            node: self.nodes,
            name: None,
            initializer: self.initializers,
            input: self.inputs,
            output: self.outputs,
            quantization_annotation: None,
        })
    }

    fn add_value_node(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        dims: Vec<usize>,
    ) -> Result<usize, GraphBuildError> {
        let name = name.into();
        if self.name_to_id.contains_key(&name) {
            return Err(GraphBuildError::DuplicateName(name));
        }

        let id = self.nodes.len();
        let mut node = Node::new(id);
        node.set_name(name.clone());
        node.set_op(Op::NoOp);
        node.set_type_value(TypeValue::Tensor {
            ty: dtype.into(),
            dims,
            has_dynamic_dims: false,
        });
        self.nodes.push(node);
        self.name_to_id.insert(name, id);
        Ok(id)
    }

    fn resolve_names(&self, names: &[&str]) -> Result<Vec<usize>, GraphBuildError> {
        names
            .iter()
            .map(|name| {
                self.name_to_id
                    .get(*name)
                    .copied()
                    .ok_or_else(|| GraphBuildError::UnknownName((*name).to_string()))
            })
            .collect()
    }
}

/// Fluent handle returned by [`GraphBuilder::op`] for attaching attributes.
pub struct OpBuilder<'a> {
    builder: &'a mut GraphBuilder,
    op_id: usize,
}

impl fmt::Debug for OpBuilder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpBuilder")
            .field("op_id", &self.op_id)
            .finish()
    }
}

impl OpBuilder<'_> {
    /// Attach a named attribute to the op created by the preceding `op` call.
    pub fn attr(self, name: impl Into<String>, value: AttributeType) -> Self {
        let attr = Attribute {
            name: name.into(),
            ref_attr_name: None,
            ty: value,
            doc_string: None,
        };
        let node = &mut self.builder.nodes[self.op_id];
        match &mut node.attribute {
            Some(attrs) => attrs.push(attr),
            None => node.attribute = Some(vec![attr]),
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn builds_simple_add_graph() {
        let mut g = GraphBuilder::new();
        g.input("a", DataType::Float, [2, 2]).unwrap();
        g.input("b", DataType::Float, [2, 2]).unwrap();
        g.value("c", DataType::Float, [2, 2]).unwrap();
        g.op(Op::Add, &["a", "b"], &["c"]).unwrap();
        g.output("c").unwrap();
        let graph = g.build().unwrap();

        assert_eq!(graph.input.len(), 2);
        assert_eq!(graph.output, vec![2]);
        assert!(graph.initializer.is_empty());

        let add = graph
            .node
            .iter()
            .find(|n| n.op() == Op::Add)
            .expect("add op");
        assert_eq!(add.input.as_deref(), Some([0, 1].as_slice()));
        assert_eq!(add.output.as_deref(), Some([2].as_slice()));
        // Value node records its producer, matching the ONNX importer.
        assert_eq!(graph.node[2].input.as_deref(), Some([3].as_slice()));
    }

    #[test]
    fn builds_graph_with_constant_and_attrs() {
        let mut g = GraphBuilder::new();
        g.input("x", DataType::Float, [2, 3]).unwrap();
        g.constant("bias", Tensor::from_vec([2], vec![1.0f32, 2.0]).unwrap())
            .unwrap();
        g.value("y", DataType::Float, [2, 1]).unwrap();
        g.op(Op::ReduceMean, &["x"], &["y"])
            .unwrap()
            .attr("keepdims", AttributeType::Int(1))
            .attr("axes", AttributeType::Ints(vec![-1]));
        g.value("z", DataType::Float, [2, 1]).unwrap();
        g.op(Op::Add, &["y", "bias"], &["z"]).unwrap();
        g.output("z").unwrap();
        let graph = g.build().unwrap();

        assert_eq!(graph.initializer.len(), 1);
        let bias = graph.initializer.values().next().unwrap();
        assert_eq!(bias.name.as_deref(), Some("bias"));
        assert_eq!(bias.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);

        let reduce = graph
            .node
            .iter()
            .find(|n| n.op() == Op::ReduceMean)
            .unwrap();
        let attrs = reduce.attribute.as_ref().unwrap();
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].name, "keepdims");
        assert_eq!(attrs[0].int(), Some(1));
    }

    #[test]
    fn rejects_duplicate_names() {
        let mut g = GraphBuilder::new();
        g.input("a", DataType::Float, [1]).unwrap();
        let err = g.value("a", DataType::Float, [1]).unwrap_err();
        assert_eq!(err, GraphBuildError::DuplicateName("a".into()));
    }

    #[test]
    fn rejects_unknown_references() {
        let mut g = GraphBuilder::new();
        g.input("a", DataType::Float, [1]).unwrap();
        g.value("c", DataType::Float, [1]).unwrap();
        let err = g.op(Op::Add, &["a", "missing"], &["c"]).unwrap_err();
        assert_eq!(err, GraphBuildError::UnknownName("missing".into()));

        let err = g.output("nope").unwrap_err();
        assert_eq!(err, GraphBuildError::UnknownName("nope".into()));
    }

    #[test]
    fn rejects_output_not_produced() {
        let mut g = GraphBuilder::new();
        g.input("a", DataType::Float, [1]).unwrap();
        g.value("orphan", DataType::Float, [1]).unwrap();
        g.output("orphan").unwrap();
        let err = g.build().unwrap_err();
        assert_eq!(err, GraphBuildError::OutputNotProduced("orphan".into()));
    }

    #[test]
    fn rejects_empty_outputs() {
        let mut g = GraphBuilder::new();
        g.input("a", DataType::Float, [1]).unwrap();
        let err = g.build().unwrap_err();
        assert_eq!(err, GraphBuildError::NoOutputs);
    }

    #[test]
    fn allows_input_passthrough_output() {
        let mut g = GraphBuilder::new();
        g.input("a", DataType::Float, [1]).unwrap();
        g.output("a").unwrap();
        let graph = g.build().unwrap();
        assert_eq!(graph.input, graph.output);
    }

    #[test]
    fn rejects_noop_and_empty_op_outputs() {
        let mut g = GraphBuilder::new();
        g.value("a", DataType::Float, [1]).unwrap();
        assert_eq!(
            g.op(Op::NoOp, &[], &["a"]).unwrap_err(),
            GraphBuildError::InvalidOp(Op::NoOp)
        );
        assert_eq!(
            g.op(Op::Add, &["a"], &[]).unwrap_err(),
            GraphBuildError::OpMissingOutputs
        );
    }
}
