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
    roots: HashSet<NonNull<Node<D>>>,
}

impl<D> GraphBuilder<D>
where
    D: Device,
{
    pub fn new(device: D) -> Self {
        Self {
            nodes: HashMap::new(),
            roots: HashSet::new(),
        }
    }

    pub fn insert(&mut self, id: Vec<u8, D::Allocator>, node: Node<D>) -> Option<NonNull<Node<D>>> {
        let ptr = NonNull::new(Box::into_raw(Box::new(node))).expect("dada");
        self.nodes.insert(id, ptr)
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
        self.roots.insert(dst);
        Ok(())
    }

    pub fn build(self) -> Graph<D> {
        Graph {
            nodes: self.nodes,
            initializers: self.roots,
        }
    }
}

pub struct Graph<D: Device> {
    nodes: HashMap<Vec<u8, D::Allocator>, NonNull<Node<D>>>,
    initializers: HashSet<NonNull<Node<D>>>,
}

impl<D> Graph<D>
where
    D: Device,
{
    pub fn forward(&mut self) -> Result<()> {
        for mut node in self.initializers.iter().copied() {
            unsafe {
                node.as_mut().execute();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::device::{CpuDevice, Device, EncodedTensor, MockTensor};
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
            (),
            device.allocator(),
        );
        let tensor_b = Node::<CpuDevice>::new(
            device
                .new_tensor(EncodedTensor {
                    op: false,
                    value: 1,
                })
                .unwrap(),
            (),
            device.allocator(),
        );
        let tensor_c = Node::<CpuDevice>::new(
            device
                .new_tensor(EncodedTensor { op: true, value: 0 })
                .unwrap(),
            (),
            device.allocator(),
        );
        builder.insert("a".to_string().into_bytes(), tensor_a);
        builder.insert("b".to_string().into_bytes(), tensor_b);
        builder.insert("c".to_string().into_bytes(), tensor_c);
        builder
            .link(&"c".to_string().into_bytes(), &"a".to_string().into_bytes())
            .unwrap();
        builder
            .link(&"c".to_string().into_bytes(), &"b".to_string().into_bytes())
            .unwrap();
        let mut graph = builder.build();
        graph.forward().unwrap();
        let c_ = graph.nodes.get(&"c".to_string().into_bytes()).unwrap();
        let node = graph.initializers.get(c_).unwrap();
        assert_eq!(
            unsafe { node.as_ref().tensor() },
            &MockTensor { value: Some(2) }
        );
    }
}
