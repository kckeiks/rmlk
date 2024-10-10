use crate::args::Args;
use crate::onnx::{Category, NodeInfo, NodeWithMetadata};
use clap::Parser;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_ir::{GraphProto, ModelProto};
use std::fs;

mod args;
mod onnx;

fn main() {
    let args = Args::parse();
    let path = args.path;

    let model = fs::read(path).expect("bad");
    let mut reader = BytesReader::from_bytes(&model);
    let model_proto = ModelProto::from_reader(&mut reader, &model).unwrap();
    let node = display_node(model_proto.graph.unwrap(), "input").unwrap();

    println!(
        "{:?}",
        node
    );
    // Todo: add function instead of implementing TryInto.
    // let model: Model = model_proto.try_into().unwrap();
}

fn display_node<'a>(graph_proto: GraphProto<'a>, name: &str) -> anyhow::Result<Option<NodeWithMetadata<'a>>> {
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
