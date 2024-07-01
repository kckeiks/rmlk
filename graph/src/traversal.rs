use crate::graph::{GraphError, Result};
use crate::{Node, Op};
use bit_set::BitSet;
use std::alloc::Allocator;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

pub struct GraphNode {
    id: usize,
    /// Counts how many different subtrees need this node as a dependency.
    deps: AtomicUsize,
    /// Output source node.
    src: usize,
    /// Inputs into these nodes.
    inputs: Vec<NonNull<GraphNode>>,
    /// Outputs into these nodes.
    outputs: Vec<NonNull<GraphNode>>,
}

// Todo: we should think about making the graph traversal deterministic here and anywhere else.
// Depth-first search.
pub fn compute_order<A: Allocator + Clone>(
    nodes: &[Node<A>],
    outputs: &[usize],
    alloc: A,
) -> Result<(Vec<usize, A>, Vec<usize, A>)> {
    // Sink nodes or nodes without dependencies.
    let mut sinks = Vec::new_in(alloc.clone());
    // Nodes that represent operations and thus have dependencies.
    let mut operations = Vec::new_in(alloc.clone());

    let mut buf = Vec::with_capacity_in(nodes.len(), alloc.clone());
    // Todo: We cannot configure the allocator in bitset.
    let mut on_path = BitSet::with_capacity(nodes.len());

    for output in outputs {
        buf.push(*output);
        while let Some(next) = buf.pop() {
            if on_path.contains(next) {
                log::debug!("loop detected");
                return Err(GraphError::LoopDetected);
            }

            let node = nodes.get(next).ok_or(GraphError::InvalidTensor)?;
            if node.inputs().is_empty() {
                sinks.push(next);
            } else {
                buf.extend(node.inputs().iter());
                operations.push(next);
            }

            on_path.insert(next);
        }
        buf.clear();
        on_path.clear();
    }

    Ok((sinks, operations))
}
