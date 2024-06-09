use crate::device::Device;
use crate::node::Node;
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug)]
pub enum GraphError {
    InvalidTensor,
    TensorNotFound,
}

pub struct GraphBuilder<D: Device> {
    nodes: HashMap<Vec<u8, D::Allocator>, NonNull<Node<D>>>,
    input: HashSet<Vec<u8, D::Allocator>>,
    outputs: HashSet<Vec<u8, D::Allocator>>,
}

impl<D> GraphBuilder<D>
where
    D: Device,
{
    pub fn new(_device: D) -> Self {
        Self {
            nodes: HashMap::new(),
            input: HashSet::new(),
            outputs: HashSet::new(),
        }
    }

    pub fn insert(&mut self, id: Vec<u8, D::Allocator>, node: Node<D>) -> Option<NonNull<Node<D>>> {
        let ptr = NonNull::new(Box::into_raw(Box::new(node))).expect("dada");
        self.nodes.insert(id, ptr)
    }

    pub fn insert_input(&mut self, id: Vec<u8, D::Allocator>) {
        self.input.insert(id.clone());
    }

    pub fn insert_output(&mut self, id: Vec<u8, D::Allocator>) {
        self.outputs.insert(id.clone());
    }

    pub fn get(&self, id: &Vec<u8, D::Allocator>) -> Option<NonNull<Node<D>>> {
        self.nodes.get(id).copied()
    }

    pub fn link(&mut self, dst: &Vec<u8, D::Allocator>, src: &Vec<u8, D::Allocator>) -> Result<()> {
        let mut dst = self
            .nodes
            .get(dst)
            .copied()
            .ok_or(GraphError::InvalidTensor)?;
        let mut src = self
            .nodes
            .get(src)
            .copied()
            .ok_or(GraphError::InvalidTensor)?;
        unsafe {
            dst.as_mut().input_link(src);
            src.as_mut().output_link(dst);
        }
        Ok(())
    }

    pub fn build(self) -> Graph<D> {
        Graph {
            inputs: self.input,
            outputs: self.outputs,
            nodes: self.nodes,
        }
    }
}

pub struct Graph<D: Device> {
    inputs: HashSet<Vec<u8, D::Allocator>>,
    outputs: HashSet<Vec<u8, D::Allocator>>,
    nodes: HashMap<Vec<u8, D::Allocator>, NonNull<Node<D>>>,
}

impl<D> Graph<D>
where
    D: Device,
{
    pub fn outputs(&self) -> impl Iterator<Item = NonNull<Node<D>>> + '_ {
        self.nodes
            .iter()
            .filter(|(k, v)| self.outputs.contains(*k))
            .map(|(_, v)| *v)
    }

    pub fn forward(&mut self) -> Result<()> {
        for (id, mut node) in self.nodes.iter_mut() {
            if self.inputs.contains(id) {
                continue;
            }

            unsafe {
                node.as_mut().execute().map_err(|_| GraphError::InvalidTensor)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::device::{CpuDevice, CpuTensor, Device, EncodedTensor};
    use crate::graph::GraphBuilder;
    use crate::node::Node;

    #[test]
    fn test_simple() {
        let device = CpuDevice;
        let mut builder = GraphBuilder::new(device.clone());
        let tensor_a = Node::<CpuDevice>::new(
            device
                .new_tensor(EncodedTensor {
                    op: false,
                    value: 1,
                })
                .unwrap(),
            device.allocator(),
        );
        let tensor_b = Node::<CpuDevice>::new(
            device
                .new_tensor(EncodedTensor {
                    op: false,
                    value: 1,
                })
                .unwrap(),
            device.allocator(),
        );
        let tensor_c = Node::<CpuDevice>::new(
            device
                .new_tensor(EncodedTensor { op: true, value: 0 })
                .unwrap(),
            device.allocator(),
        );
        builder.insert("a".to_string().into_bytes(), tensor_a);
        builder.insert_input("a".to_string().into_bytes());
        builder.insert("b".to_string().into_bytes(), tensor_b);
        builder.insert_input("b".to_string().into_bytes());
        builder.insert("c".to_string().into_bytes(), tensor_c);
        builder.insert_output("c".to_string().into_bytes());
        builder
            .link(&"c".to_string().into_bytes(), &"a".to_string().into_bytes())
            .unwrap();
        builder
            .link(&"c".to_string().into_bytes(), &"b".to_string().into_bytes())
            .unwrap();
        let mut graph = builder.build();
        graph.forward().unwrap();
        let len = graph.outputs().count();
        assert_eq!(len, 1);
        for node in graph.outputs() {
            let tensor = unsafe { node.as_ref().tensor() };
            assert_eq!(
                tensor,
                &CpuTensor {
                    op: Some(()),
                    value: 2
                }
            );
        }
    }
}
