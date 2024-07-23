use crate::graph::{Graph, GraphError};
use crate::node::Node;
use crate::traversal;
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, GraphError>;

// Todo: Maybe use `hashbrown` map for these maps.
pub struct GraphBuilder {
    nodes: Vec<Node>,
    initializers: Vec<rmlk_ir::Tensor>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    /// Maps a node's name to its ID or its source's ID.
    ///
    /// This is used to map the name of a node to its ID
    /// or an output to its source node.
    name_to_node_id: HashMap<String, usize>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            name_to_node_id: HashMap::new(),
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn add_node(&mut self, node: Node) -> Result<usize> {
        let id = self.nodes.len();
        self.nodes.push(node);
        Ok(id)
    }

    pub fn add_input(&mut self, node: Node) -> Result<usize> {
        let id = self.add_node(node)?;
        self.inputs.push(id);
        Ok(id)
    }

    pub fn add_output(&mut self, node: Node) -> Result<usize> {
        let id = self.add_node(node)?;
        self.outputs.push(id);
        Ok(id)
    }

    pub fn add_initial_tensor(&mut self, tensor: rmlk_ir::Tensor) -> usize {
        let id = self.initializers.len();
        self.initializers.push(tensor);
        id
    }

    pub fn get_node_id(&self, name: &String) -> Option<usize> {
        self.name_to_node_id.get(name).copied()
    }

    pub fn insert_name_to_id(&mut self, name: String, id: usize) -> Option<usize> {
        self.name_to_node_id.insert(name, id)
    }

    pub fn build(self) -> Result<Graph> {
        // let (_, _plan) = traversal::compute_order(self.nodes.as_slice(), self.outputs.as_slice())?;

        Ok(Graph::new(
            self.initializers,
            self.inputs,
            self.nodes,
            self.outputs,
        ))
    }
}
