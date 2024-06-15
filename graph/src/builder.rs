use crate::device::Device;
use crate::graph::Graph;
use crate::node::Node;
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, BuilderError>;

pub enum BuilderError {
    Unknown,
}

pub struct GraphBuilder<D: Device> {
    /// The device.
    device: D,
    /// All the nodes in the graph.
    nodes: Vec<Node<D>, D::Allocator>,
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

    pub fn add_node(&mut self, node: Node<D>) -> Result<usize> {
        let id = self.nodes.len();
        self.nodes.push(node);
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

    pub fn store_id_by_name(&mut self, name: String, id: usize) -> Option<usize> {
        self.name_to_id.insert(name, id)
    }

    pub fn build(self) -> Result<Graph<D>> {
        for output in self.outputs {
            let node = self.nodes.get(output).expect("TODO");
            for input in node.inputs() {}
        }
    }
}
