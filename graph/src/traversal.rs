use crate::graph::{GraphError, Result};
use crate::Node;
use rmlk_ir::{Category, NodeWithMetadata, NodeWithValue};
use rmlk_ir::{GraphProto, Op};
use std::collections::HashSet;

#[derive(Debug)]
pub enum TraversalError {
    Unknown,
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
    fn check_node(
        &mut self,
        node: NodeWithMetadata<'a>,
    ) -> std::result::Result<bool, TraversalError>;
}

pub fn visit<'a, T>(
    graph_proto: GraphProto<'a>,
    traverser: &mut T,
) -> std::result::Result<(), TraversalError>
where
    T: OnnxGraphTraverser<'a>,
{
    for initializer in graph_proto.initializer {
        let node = NodeWithValue::try_from(initializer)
            .map_err(|_| GraphError::LoopDetected)
            .unwrap();
        if traverser.check_node(NodeWithMetadata {
            category: Category::Initializer,
            node_with_value: node,
        })? {
            return Ok(());
        }
    }

    for input in graph_proto.input {
        let node = NodeWithValue::from(input);
        if traverser.check_node(NodeWithMetadata {
            category: Category::Input,
            node_with_value: node,
        })? {
            return Ok(());
        }
    }

    for output in graph_proto.output {
        let node = NodeWithValue::from(output);
        if traverser.check_node(NodeWithMetadata {
            category: Category::Output,
            node_with_value: node,
        })? {
            return Ok(());
        }
    }

    for node in graph_proto.node {
        let node = NodeWithValue::try_from(node)
            .map_err(|_| GraphError::LoopDetected)
            .unwrap();
        if traverser.check_node(NodeWithMetadata {
            category: Category::InnerNode,
            node_with_value: node,
        })? {
            return Ok(());
        }
    }

    Ok(())
}
