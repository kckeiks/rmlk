use crate::args::{Args, Command};
use anyhow::anyhow;
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_graph::{OnnxGraphTraverser, TraversalError};
use rmlk_ir::ModelProto;
use rmlk_ir::NodeWithMetadata;
use std::fs;

mod args;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.cmd {
        Command::Find { path, target } => {
            let model = fs::read(path)?;
            let mut reader = BytesReader::from_bytes(&model);
            let model_proto = ModelProto::from_reader(&mut reader, &model)?;
            let mut traverser = FindNode {
                target: target.as_ref(),
                node: None,
            };
            rmlk_graph::visit_onnx(
                model_proto
                    .graph
                    .ok_or(anyhow!("the model does not have a graph"))?,
                &mut traverser,
            )
            .map_err(|e| anyhow!("an error ocurred while traversing the onnx graph: {e:?}"))?;
            match traverser.node {
                Some(node) => println!("{:?}", node),
                None => println!("we did not find a node with the name `{target}`"),
            }
        }
    }

    Ok(())
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
