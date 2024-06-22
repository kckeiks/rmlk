use crate::device::Device;
use crate::graph::{Graph, GraphError};
use crate::node::Node;
use crate::traversal;
use std::collections::HashMap;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, GraphError>;

pub struct GraphBuilder<D: Device> {
    /// The device.
    device: D,
    /// All the nodes in the graph.
    nodes: Vec<Arc<Node<D>, D::Allocator>, D::Allocator>,
    /// Inputs of the graph.
    input: Vec<usize, D::Allocator>,
    /// Outputs of the graph.
    outputs: Vec<usize, D::Allocator>,
    /// Name to node.
    name_to_id: HashMap<String, usize>,
}

impl<D> GraphBuilder<D>
where
    D: Device,
{
    pub fn new(device: D) -> Self {
        Self {
            nodes: Vec::new_in(device.allocator()),
            input: Vec::new_in(device.allocator()),
            outputs: Vec::new_in(device.allocator()),
            name_to_id: HashMap::new(),
            device,
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&Arc<Node<D>>> {
        self.nodes.get(id)
    }

    pub fn add_node(&mut self, node: Node<D>) -> Result<usize> {
        let id = self.nodes.len();
        self.nodes.push(Arc::new_in(node, self.device.allocator()));
        Ok(id)
    }

    pub fn add_input(&mut self, node: Node<D>) -> Result<usize> {
        let id = self.add_node(node)?;
        self.input.push(id);
        Ok(id)
    }

    pub fn add_output(&mut self, node: Node<D>) -> Result<usize> {
        let id = self.add_node(node)?;
        self.outputs.push(id);
        Ok(id)
    }

    pub fn get_store_id_by_name(&mut self, name: &String) -> Option<usize> {
        self.name_to_id.get(name).copied()
    }

    pub fn store_id_by_name(&mut self, name: String, id: usize) -> Option<usize> {
        self.name_to_id.insert(name, id)
    }

    pub fn build(self) -> Result<Graph<D>> {
        let (_, operations) = traversal::compute_order(
            self.nodes.as_slice(),
            self.outputs.as_slice(),
            self.device.allocator(),
        )?;

        Ok(Graph::new(
            self.device,
            self.nodes,
            self.outputs,
            operations,
        ))
    }
}
