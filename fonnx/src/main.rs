use crate::args::{Args, Command};
use anyhow::anyhow;
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_graph::{OnnxGraphTraverser, TraversalError};
use rmlk_ir::onnx::{
    Category, ModelProto, NodeProto, NodeWithMetadata, NodeWithValue, TensorProto, ValueInfoProto,
};
use std::fs;

mod args;
mod transform;

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
    fn check_input(&mut self, input: ValueInfoProto<'a>) -> Result<bool, TraversalError> {
        match &input.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: None,
                        tensor: None,
                        value: Some(input),
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn check_output(&mut self, output: ValueInfoProto<'a>) -> Result<bool, TraversalError> {
        match &output.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: None,
                        tensor: None,
                        value: Some(output),
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> Result<bool, TraversalError> {
        match &initializer.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: None,
                        tensor: Some(initializer),
                        value: None,
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn check_inner_node(&mut self, node: NodeProto<'a>) -> Result<bool, TraversalError> {
        match &node.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: Some(node),
                        tensor: None,
                        value: None,
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
