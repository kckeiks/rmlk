use rmlk_schema::onnx::{NodeProto, TensorProto, ValueInfoProto};
use rmlk_schema::onnx_import::{OnnxGraphTraverser, Result};

pub struct List {
    pub node: Vec<String>,
}

impl<'a> OnnxGraphTraverser<'a> for List {
    fn check_input(&mut self, v: ValueInfoProto<'a>) -> Result<bool> {
        self.node.push(format!(
            "input={:?}[shape={:?}]",
            v.name,
            v.type_pb.unwrap().value
        ));
        Ok(false)
    }

    fn check_output(&mut self, v: ValueInfoProto<'a>) -> Result<bool> {
        self.node.push(format!(
            "output={:?}[info={:?}]",
            v.name,
            v.type_pb.unwrap().value
        ));
        Ok(false)
    }

    fn check_initializer(&mut self, v: TensorProto<'a>) -> Result<bool> {
        self.node.push(format!(
            "{:?}: shape={:?}, dtype={:?}, elements={}",
            v.name.clone().unwrap(),
            v.dims,
            v.data_type,
            v.dims.iter().copied().product::<i64>()
        ));
        Ok(false)
    }

    fn check_inner_node(&mut self, node: NodeProto<'a>) -> Result<bool> {
        let inputs = node.input.clone();
        let outputs = node.output.clone();
        self.node.push(format!(
            "node={:?}[inputs={:?}][outputs={:?}]",
            node.name, inputs, outputs
        ));
        Ok(false)
    }
}
