mod args;
mod find;
mod traverse;

use anyhow::anyhow;
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_ir::onnx::ModelProto;
use std::fs;

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
    }

    Ok(())
}
