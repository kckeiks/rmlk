use crate::graph::{GraphError, Result};
use crate::Node;
use bit_set::BitSet;

// Todo: we should think about making the graph traversal deterministic here and anywhere else.
// Depth-first search.
pub fn compute_order(nodes: &[Node], outputs: &[usize]) -> Result<(Vec<usize>, Vec<usize>)> {
    // Sink nodes or nodes without dependencies.
    let mut sinks = Vec::new();
    // Nodes that represent operations and thus have dependencies.
    let mut operations = Vec::new();

    let mut buf = Vec::with_capacity(nodes.len());
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
