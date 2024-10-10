use crate::args::{Args, Command};
use crate::onnx::{Category, NodeInfo, NodeWithMetadata};
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_ir::{GraphProto, ModelProto};
use std::fs;

mod args;
mod onnx;

fn main() {
    let args = Args::parse();
    let path = match args.cmd {
        Command::Find { path, target } => {
            let model = fs::read(path).expect("bad");
            let mut reader = BytesReader::from_bytes(&model);
            let model_proto = ModelProto::from_reader(&mut reader, &model).unwrap();
            let mut traverser = FindNode {
                target: target.as_ref(),
                node: None,
            };

            visit(model_proto.graph.unwrap(), &mut traverser).unwrap();

            // let node = display_node(model_proto.graph.unwrap(), "/layer1/layer1.0/conv1/Conv").unwrap();

            println!("{:?}", traverser.node.unwrap());
            // Todo: add function instead of implementing TryInto.
            // let model: Model = model_proto.try_into().unwrap();
        }
    };
}

pub struct FindNode<'a> {
    pub target: &'a str,
    pub node: Option<NodeWithMetadata<'a>>,
}

impl<'a> OnnxGraphTraverser<'a> for FindNode<'a> {
    fn check_node(&mut self, node: NodeWithMetadata<'a>) -> anyhow::Result<bool> {
        match &node.node.name {
            Some(name) if name == self.target => {
                self.node = Some(node);
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

pub trait OnnxGraphTraverser<'a> {
    fn check_node(&mut self, node: NodeWithMetadata<'a>) -> anyhow::Result<bool>;
}

fn visit<'a, T>(graph_proto: GraphProto<'a>, traverser: &mut T) -> anyhow::Result<()>
where
    T: OnnxGraphTraverser<'a>,
{
    for initializer in graph_proto.initializer {
        let node = NodeInfo::try_from(initializer)?;
        if traverser.check_node(NodeWithMetadata {
            category: Category::Initializer,
            node,
        })? {
            return Ok(());
        }
    }

    for input in graph_proto.input {
        let node = NodeInfo::try_from(input)?;
        if traverser.check_node(NodeWithMetadata {
            category: Category::Input,
            node,
        })? {
            return Ok(());
        }
    }

    for output in graph_proto.output {
        let node = NodeInfo::try_from(output)?;
        if traverser.check_node(NodeWithMetadata {
            category: Category::Output,
            node,
        })? {
            return Ok(());
        }
    }

    for node in graph_proto.node {
        let node = NodeInfo::from(node);
        if traverser.check_node(NodeWithMetadata {
            category: Category::InnerNode,
            node,
        })? {
            return Ok(());
        }
    }

    Ok(())
}

fn display_node<'a>(
    graph_proto: GraphProto<'a>,
    name: &str,
) -> anyhow::Result<Option<NodeWithMetadata<'a>>> {
    for initializer in graph_proto.initializer {
        if let Some(initializer_name) = &initializer.name {
            if initializer_name == name {
                let node = NodeInfo::try_from(initializer)?;
                return Ok(Some(NodeWithMetadata {
                    category: Category::Initializer,
                    node,
                }));
            }
        }
    }

    for input in graph_proto.input {
        if let Some(input_name) = &input.name {
            if input_name == name {
                let node = NodeInfo::try_from(input)?;
                return Ok(Some(NodeWithMetadata {
                    category: Category::Input,
                    node,
                }));
            }
        }
    }

    for output in graph_proto.output {
        if let Some(output_name) = &output.name {
            if output_name == name {
                let node = NodeInfo::try_from(output)?;
                return Ok(Some(NodeWithMetadata {
                    category: Category::Output,
                    node,
                }));
            }
        }
    }

    for node in graph_proto.node {
        if let Some(node_name) = &node.name {
            if node_name == name {
                let node = NodeInfo::from(node);
                return Ok(Some(NodeWithMetadata {
                    category: Category::InnerNode,
                    node,
                }));
            }
        }
    }

    Ok(None)
}

// // 1.
// // Todo: build tool to look for a node with a given name and display its protobuf implementation.
// // Todo: Maybe implementa a visitor trait and a function to traverse the graph while assigining ids.
// // Ids are not generic and this will be consistent across different visitor implementation.
// // THis trait-based method can also perform the spec validation as well.
// // Can also support other look up like unsupported ops and datatypes, etc.
// // 2.
// // Design encoding scheme for deployment files.
// pub fn process_for_debug(graph: &Graph) -> anyhow::Result<()> {
//     for tensor in graph.initializer {
//         match tensor.data_type {
//             DataType::Undefined => {
//                 println!("")
//             }
//             DataType::Int8 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Float => {
//                 res.float_data = u8_to_f32_vec(data.as_slice())?;
//             }
//             DataType::Double => {
//                 res.double_data = u8_to_f64_vec(data.as_slice())?;
//             }
//             DataType::Uint8 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Uint16 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Int16 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Int32 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Uint32 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Int64 => {
//                 res.int64_data = u8_to_i64_vec(data.as_slice())?;
//             }
//             DataType::Uint64 => {
//                 res.uint64_data = u8_to_u64_vec(data.as_slice())?;
//             }
//             DataType::String => {}
//             DataType::Bool => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Float16 => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Bfloat16 => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Complex64 => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Complex128 => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Float8E4M3FN => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Float8E4M3FNUZ => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Float8E5M2 => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Float8E5M2FNUZ => {
//                 return Err(Error::NotSupported);
//             }
//             DataType::Uint4 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//             DataType::Int4 => {
//                 res.int32_data = u8_to_i32_vec(data.as_slice())?;
//             }
//         }
//     }
//
//     Ok(())
// }
