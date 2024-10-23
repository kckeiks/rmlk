use crate::graph::{GraphError, Result};
use crate::Node;
use rmlk_ir::{
    Category, Graph, NodeProto, NodeWithMetadata, NodeWithValue, TensorProto, ValueInfoProto,
};
use rmlk_ir::{GraphProto, Op};
use std::collections::HashSet;

#[derive(Debug)]
pub enum TraversalError {
    Unknown,
    MissingValue,
    MissingType,
    TransformationFailed,
    InvalidTensor,
    InvalidValue(String),
    InvalidInnerNode,
}

// Todo: we should think about making the graph traversal deterministic here and anywhere else.
// Depth-first search.
pub fn compute_order(nodes: &[Node], outputs: &[usize]) -> Result<(Vec<usize>, Vec<usize>)> {
    // Sink nodes or nodes without dependencies.
    let mut sinks = Vec::new();
    // Nodes that represent operations and thus have dependencies.
    let mut operations = Vec::new();

    let mut buf = Vec::with_capacity(nodes.len());
    // Todo: We cannot configure the allocator in bitset.
    let mut on_path = HashSet::with_capacity(nodes.len());

    // Todo: use BitSet?
    let mut already_seen = HashSet::new();

    for output in outputs {
        buf.push(*output);
        while let Some(next) = buf.pop() {
            if on_path.contains(&next) {
                println!("loop detected {next}");
                println!("{:?}", on_path);
                return Err(GraphError::LoopDetected);
            }

            if already_seen.contains(&next) {
                continue;
            }

            let node = nodes.get(next).ok_or(GraphError::InvalidTensor)?;
            if node.inputs().is_empty() {
                // println!("Sink: {:?} {next}", node.op());
                sinks.push(next);
                on_path.clear();
            } else {
                // println!("Not Sink: {:?} {next} -> {:?}", node.op(), node.inputs());
                if matches!(node.op(), Op::Const) {
                    // println!("Initializer {next}");
                }

                if !matches!(node.op(), Op::Const) {
                    buf.extend(node.inputs().iter());
                }
                // println!("Inserting: {} {:?}", next, node.op());
                operations.push(next);
                already_seen.insert(next);
                on_path.insert(next);
            }
        }

        buf.clear();
        on_path.clear();
    }

    Ok((sinks, operations))
}

pub trait OnnxGraphTraverser<'a> {
    fn check_input(
        &mut self,
        input: ValueInfoProto<'a>,
    ) -> std::result::Result<bool, TraversalError>;
    fn check_output(
        &mut self,
        output: ValueInfoProto<'a>,
    ) -> std::result::Result<bool, TraversalError>;
    fn check_initializer(
        &mut self,
        initializer: TensorProto<'a>,
    ) -> std::result::Result<bool, TraversalError>;
    fn check_inner_node(
        &mut self,
        node: NodeProto<'a>,
    ) -> std::result::Result<bool, TraversalError>;
}

pub fn visit_onnx<'a, T>(
    graph_proto: GraphProto<'a>,
    traverser: &mut T,
) -> std::result::Result<(), TraversalError>
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

pub trait GraphTraverser {
    fn check_input(
        &mut self,
        input: rmlk_ir::ValueInfo,
    ) -> std::result::Result<bool, TraversalError>;
    fn check_output(
        &mut self,
        output: rmlk_ir::ValueInfo,
    ) -> std::result::Result<bool, TraversalError>;
    fn check_initializer(
        &mut self,
        initializer: rmlk_ir::Tensor,
    ) -> std::result::Result<bool, TraversalError>;
    fn check_inner_node(
        &mut self,
        node: rmlk_ir::Node,
    ) -> std::result::Result<bool, TraversalError>;
}

pub fn visit_graph<T>(graph: Graph, traverser: &mut T) -> std::result::Result<(), TraversalError>
where
    T: GraphTraverser,
{
    for initializer in graph.initializer {
        if traverser.check_initializer(initializer)? {
            return Ok(());
        }
    }

    for input in graph.input {
        if traverser.check_input(input)? {
            return Ok(());
        }
    }

    for output in graph.output {
        if traverser.check_output(output)? {
            return Ok(());
        }
    }

    for node in graph.node {
        if traverser.check_inner_node(node)? {
            return Ok(());
        }
    }

    Ok(())
}
