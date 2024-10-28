mod args;
mod find;
mod transform;
mod traverse;

use anyhow::anyhow;
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_schema::onnx::ModelProto;
use std::fs;

use crate::transform::graph_from_onnx_proto;
use args::{Args, Command};
use find::FindNode;

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
            traverse::visit_onnx(
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
        Command::Transform { path } => {
            let model = fs::read(path)?;
            let mut reader = BytesReader::from_bytes(&model);
            let model_proto = ModelProto::from_reader(&mut reader, &model)?;
            let compute_graph = graph_from_onnx_proto(model_proto)?;
            let serialized_model = bincode::serialize(&compute_graph)?;
            fs::write("resnet34.rmlk", serialized_model)?;
        }
    }

    Ok(())
}
