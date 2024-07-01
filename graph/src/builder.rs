use crate::graph::{Graph, GraphError};
use crate::node::Node;
use crate::traversal;
use std::alloc::Allocator;
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, GraphError>;

pub struct GraphBuilder<A: Allocator> {
    alloc: A,
    nodes: Vec<Node<A>, A>,
    initializers: Vec<rmlk_hir::Tensor, A>,
    inputs: Vec<usize, A>,
    outputs: Vec<usize, A>,
    name_to_id: HashMap<String, usize>,
}

impl<A> GraphBuilder<A>
where
    A: Allocator + Clone,
{
    pub fn new(alloc: A) -> Self {
        Self {
            nodes: Vec::new_in(alloc.clone()),
            initializers: Vec::new_in(alloc.clone()),
            inputs: Vec::new_in(alloc.clone()),
            outputs: Vec::new_in(alloc.clone()),
            name_to_id: HashMap::new(),
            alloc,
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&Node<A>> {
        self.nodes.get(id)
    }

    pub fn add_node(&mut self, node: Node<A>) -> Result<usize> {
        let id = self.nodes.len();
        self.nodes.push(node);
        Ok(id)
    }

    pub fn add_input(&mut self, node: Node<A>) -> Result<usize> {
        let id = self.add_node(node)?;
        self.inputs.push(id);
        Ok(id)
    }

    pub fn add_output(&mut self, node: Node<A>) -> Result<usize> {
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

    pub fn build(self) -> Result<Graph<A>> {
        let (_, plan) =
            traversal::compute_order(self.nodes.as_slice(), self.outputs.as_slice(), self.alloc)?;

        Ok(Graph::new(
            self.initializers,
            self.inputs,
            self.nodes,
            self.outputs,
        ))
    }
}
