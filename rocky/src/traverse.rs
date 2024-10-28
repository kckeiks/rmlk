use rmlk_schema::onnx::{GraphProto, NodeProto, TensorProto, ValueInfoProto};

pub type Result<T> = std::result::Result<T, TraversalError>;

#[derive(Debug)]
pub enum TraversalError {
    // Todo: Remove.
    #[allow(unused)]
    InvalidValue(String),
    InvalidInnerNode,
}

pub trait OnnxGraphTraverser<'a> {
    fn check_input(&mut self, input: ValueInfoProto<'a>) -> Result<bool>;
    fn check_output(&mut self, output: ValueInfoProto<'a>) -> Result<bool>;
    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> Result<bool>;
    fn check_inner_node(&mut self, node: NodeProto<'a>) -> Result<bool>;
}

pub fn visit_onnx<'a, T>(graph_proto: GraphProto<'a>, traverser: &mut T) -> Result<()>
where
    T: OnnxGraphTraverser<'a>,
{
    for initializer in graph_proto.initializer {
        if traverser.check_initializer(initializer)? {
            return Ok(());
        }
    }

    for input in graph_proto.input {
        if traverser.check_input(input)? {
            return Ok(());
        }
    }

    for output in graph_proto.output {
        if traverser.check_output(output)? {
            return Ok(());
        }
    }

    for node in graph_proto.node {
        if traverser.check_inner_node(node)? {
            return Ok(());
        }
    }

    Ok(())
}
