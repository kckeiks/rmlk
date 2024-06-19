use crate::device::Device;
use crate::node::Node;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug)]
pub enum GraphError {
    InvalidTensor,
    TensorNotFound,
    UnknownNode,
    LoopDetected,
    Unknown,
    NodeInUse,
    ComputationFailed,
}

pub struct Graph<D: Device> {
    device: D,
    nodes: Vec<Arc<Node<D>, D::Allocator>, D::Allocator>,
    outputs: Vec<usize, D::Allocator>,
    steps: Vec<usize, D::Allocator>,
}

impl<D> Graph<D>
where
    D: Device,
{
    pub(crate) fn new(
        device: D,
        nodes: Vec<Arc<Node<D>, D::Allocator>, D::Allocator>,
        outputs: Vec<usize, D::Allocator>,
        steps: Vec<usize, D::Allocator>,
    ) -> Self {
        Self {
            device,
            nodes,
            outputs,
            steps,
        }
    }

    pub fn outputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outputs.iter().copied()
    }

    pub fn get(&self, id: usize) -> Option<&Arc<Node<D>, D::Allocator>> {
        self.nodes.get(id)
    }

    pub fn forward(&mut self) -> Result<()> {
        for index in self.steps.iter() {
            let node = self.nodes.get(*index).ok_or(GraphError::InvalidTensor)?;
            let mut inputs = Vec::new_in(self.device.allocator());
            for id in node.inputs() {
                let dep = self
                    .nodes
                    .get(*id)
                    .cloned()
                    .ok_or(GraphError::UnknownNode)?;
                inputs.push(dep);
            }
            let node = Arc::get_mut(
                self.nodes
                    .get_mut(*index)
                    .ok_or(GraphError::InvalidTensor)?,
            )
            .ok_or(GraphError::NodeInUse)?;
            node.execute(inputs.into_iter()).map_err(|e| {
                println!("{:?}", e);
                GraphError::ComputationFailed
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::device::{CpuDevice, Device};
    use crate::{GraphBuilder, Node, Op};
    use rmlk_hir::{DataType, Tensor};

    fn hir_tensor(shape: Vec<usize>, name: String, data: Vec<f32>) -> Tensor {
        let mut tensor = Tensor::default();
        tensor.dims = shape;
        tensor.name = Some(name);
        tensor.float_data = data;
        tensor.data_type = Some(DataType::Float);
        tensor
    }

    #[test]
    fn test_graph_simple() {
        let device = CpuDevice;

        let tensor_a = hir_tensor(
            vec![3, 3],
            "a".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let a = Node::new_with_tensor(
            Op::NoOp,
            device.clone(),
            device.tensor_from_hir(tensor_a).unwrap(),
        );
        let tensor_b = hir_tensor(
            vec![3, 3],
            "b".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let b = Node::new_with_tensor(
            Op::NoOp,
            device.clone(),
            device.tensor_from_hir(tensor_b).unwrap(),
        );
        let mut c = Node::new(Op::MatMul, device.clone());

        let mut builder = GraphBuilder::new(device.clone());
        let a_id = builder.add_input(b).unwrap();
        let b_id = builder.add_input(a).unwrap();

        let mut inputs = c.inputs_mut();
        inputs.push(a_id);
        inputs.push(b_id);

        builder.add_output(c).unwrap();

        let mut graph = builder.build().unwrap();
        graph.forward().unwrap();

        for n in graph.outputs() {
            let node = graph.get(n).unwrap();
            println!("Result: {:?}", node.tensor())
        }
    }
}
