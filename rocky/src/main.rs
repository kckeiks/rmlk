mod args;
mod find;
mod list;
mod list_ops;
mod transform;
mod traverse;

use anyhow::anyhow;
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_schema::onnx::ModelProto;
use std::collections::HashSet;
use std::fs;

use crate::list::List;
use crate::list_ops::ListOps;
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
        Command::Transform { path, output } => {
            let base_url = path.parent().map(|p| p.to_path_buf());
            println!("base_url: {:?}", base_url);
            let model = fs::read(path.clone())?;
            let mut reader = BytesReader::from_bytes(&model);
            let model_proto = ModelProto::from_reader(&mut reader, &model)?;
            let compute_graph = graph_from_onnx_proto(model_proto, base_url)?;
            let serialized_model = bincode::serialize(&compute_graph)?;
            if let Some(output) = output {
                fs::write(format!("{output}.rmlk"), serialized_model)?;
            } else {
                let filename = path
                    .file_stem()
                    .expect("expecting a file stem")
                    .to_string_lossy()
                    .to_string();
                fs::write(format!("{filename}.rmlk"), serialized_model)?;
            }
        }
        Command::ListOps { path } => {
            let model = fs::read(path)?;
            let mut reader = BytesReader::from_bytes(&model);
            let model_proto = ModelProto::from_reader(&mut reader, &model)?;
            let mut traverser = ListOps {
                ops: HashSet::new(),
            };
            traverse::visit_onnx(
                model_proto
                    .graph
                    .ok_or(anyhow!("the model does not have a graph"))?,
                &mut traverser,
            )
            .map_err(|e| anyhow!("an error ocurred while traversing the onnx graph: {e:?}"))?;
            println!("{:?}", traverser.ops);
        }
        Command::List { path } => {
            let model = fs::read(path)?;
            let mut reader = BytesReader::from_bytes(&model);
            let model_proto = ModelProto::from_reader(&mut reader, &model)?;
            let mut traverser = List { node: vec![] };
            traverse::visit_onnx(
                model_proto
                    .graph
                    .ok_or(anyhow!("the model does not have a graph"))?,
                &mut traverser,
            )
            .map_err(|e| anyhow!("an error ocurred while traversing the onnx graph: {e:?}"))?;
            for node in traverser.node {
                println!("{:?}", node);
            }
            // println!("{:?}", traverser.node.len());
        }
    }

    Ok(())
}
