use crate::graph::{GraphError, Result};
use crate::Node;
use rmlk_ir::Op;
use std::collections::HashSet;

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
