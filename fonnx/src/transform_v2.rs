use rmlk_graph::{OnnxGraphTraverser, TraversalError};
use rmlk_ir::{NodeProto, TensorProto, ValueInfoProto};
use std::collections::HashMap;

pub struct GraphFromOnnxV2 {
    debug_mode: bool,
    nodes: Vec<usize>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    name_to_id: HashMap<String, u64>,
}

impl<'a> OnnxGraphTraverser<'a> for GraphFromOnnxV2 {
    fn check_input(&mut self, input: ValueInfoProto<'a>) -> Result<bool, TraversalError> {
        todo!()
    }

    fn check_output(&mut self, output: ValueInfoProto<'a>) -> Result<bool, TraversalError> {
        todo!()
    }

    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> Result<bool, TraversalError> {
        todo!()
    }

    fn check_inner_node(&mut self, node: NodeProto<'a>) -> Result<bool, TraversalError> {
        todo!()
    }
}
