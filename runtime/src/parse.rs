use crate::traverse;
use crate::traverse::GraphFromOnnx;
use rmlk_schema::onnx::GraphProto;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Device,
    MissingInputNode,
    Unknown,
}

pub fn parse_ir_graph(graph_schema: GraphProto) -> Result<rmlk_graph::Graph> {
    let mut traverser = GraphFromOnnx::default();
    traverse::visit_onnx(graph_schema, &mut traverser).unwrap();
    Ok(rmlk_graph::Graph::from(traverser))
}
