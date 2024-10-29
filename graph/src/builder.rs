use crate::graph::{Graph, GraphError};
use crate::node::Node;
use log::trace;

pub type Result<T> = std::result::Result<T, GraphError>;

pub struct GraphBuilder<T> {
    nodes: Vec<Node<T>>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl<T> GraphBuilder<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&Node<T>> {
        self.nodes.get(id)
    }

    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut Node<T>> {
        self.nodes.get_mut(id)
    }

    pub fn add_node(&mut self, node: Node<T>) -> Result<usize> {
        let id = self.nodes.len();
        self.nodes.push(node);
        Ok(id)
    }

    pub fn add_input(&mut self, node: Node<T>) -> Result<usize> {
        let id = self.add_node(node)?;
        self.inputs.push(id);
        Ok(id)
    }

    pub fn add_output(&mut self, id: usize) -> Result<usize> {
        self.outputs.push(id);
        Ok(id)
    }

    pub fn add_output_node(&mut self, node: Node<T>) -> Result<usize> {
        let id = self.add_node(node)?;
        self.outputs.push(id);
        Ok(id)
    }

    pub fn build(self) -> Result<Graph<T>> {
        for (id, n) in self.nodes.iter().enumerate() {
            let inputs = n.inputs();
            let outputs = n.outputs();
            trace!(
                "node_id={id},\
                inputs={inputs:?},\
                outputs={outputs:?}",
            );
        }
        Ok(Graph::new(self.inputs, self.nodes, self.outputs))
    }
}
