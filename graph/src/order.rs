use bit_set::BitSet;
use log::debug;
use crate::device::Device;
use crate::graph::{GraphError, Result};
use crate::Node;

// Todo: we should think about making the graph traversal deterministic here and anywhere else.
// Depth-first search.
pub fn compute_order<D: Device>(
    nodes: &[Node<D>],
    inputs: &[usize],
    outputs: &[usize],
    alloc: D::Allocator
) -> Result<Vec<usize, D::Allocator>> {
    // Todo: We cannot configure the allocator in bitset.
    let mut on_path = BitSet::with_capacity(nodes.len());
    let mut buf = Vec::with_capacity_in(nodes.len(), alloc);
    for output in outputs {
        buf.push(output);
        while !buf.is_empty() {

        }
        let node = nodes.get(*output).ok_or(GraphError::InvalidTensor)?;
        for input in node.inputs() {
            if on_path.insert(*input) {
                debug!("a loop was detected");
                return Err(GraphError::LoopDetected);
            }
        }
    }

    Ok(vec![])
}
