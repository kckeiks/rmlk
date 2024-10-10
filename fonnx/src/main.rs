use crate::args::{Args, Command};
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_graph::{OnnxGraphTraverser, TraversalError};
use rmlk_ir::ModelProto;
use rmlk_ir::NodeWithMetadata;
use std::fs;

mod args;

fn main() {
    let args = Args::parse();
    match args.cmd {
        Command::Find { path, target } => {
            let model = fs::read(path).expect("bad");
            let mut reader = BytesReader::from_bytes(&model);
            let model_proto = ModelProto::from_reader(&mut reader, &model).unwrap();
            let mut traverser = FindNode {
                target: target.as_ref(),
                node: None,
            };
            rmlk_graph::visit(model_proto.graph.unwrap(), &mut traverser).unwrap();

            println!("{:?}", traverser.node.unwrap());
        }
    }
}

pub struct FindNode<'a> {
    pub target: &'a str,
    pub node: Option<NodeWithMetadata<'a>>,
}

impl<'a> OnnxGraphTraverser<'a> for FindNode<'a> {
    fn check_node(&mut self, node: NodeWithMetadata<'a>) -> Result<bool, TraversalError> {
        match node.name() {
            Some(name) if name == self.target => {
                self.node = Some(node);
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
