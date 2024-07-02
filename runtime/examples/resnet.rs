#![feature(allocator_api)]

use std::alloc::Global;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_ir::{Model, ModelProto};
use std::fs;

fn main() {
    // let model = fs::read("/Users/acadia/Repo/Llama-2-7b-ONNX/FP32-Chat/LlamaV2_7B_FT_float32.onnx")
    //     .expect("bad");
    // let model = fs::read("/Users/acadia/Repo/notebooks/resnet34.onnx")
    //     .expect("bad");
    let model =
        fs::read("/home/mmeier/Downloads/resnet34.onnx")
            .expect("bad");
    let mut reader = BytesReader::from_bytes(&model);
    let model_proto = ModelProto::from_reader(&mut reader, &model).unwrap();

    println!(
        "onnx initializers {:?}",
        model_proto.graph.as_ref().unwrap().initializer.len()
    );
    println!(
        "onnx nodes {:?}",
        model_proto.graph.as_ref().unwrap().node.len()
    );
    println!(
        "onnx inputs {:?}",
        model_proto.graph.as_ref().unwrap().input.len()
    );
    println!(
        "onnx outputs {:?}\n",
        model_proto.graph.as_ref().unwrap().output.len()
    );

    let rmlk_model: Model = model_proto.try_into().unwrap();
    let _res = bincode::serialize(&rmlk_model).unwrap();

    let graph = rmlk_runtime::parse::parse_ir_graph(rmlk_model.graph.unwrap(), Global).unwrap();
    println!("Nodes {:?}", graph.nodes().count());
    println!("Inputs {:?}", graph.inputs().count());
    println!("Outputs {:?}", graph.outputs().count());
    println!("Initializers {:?}", graph.initializers().count());
}
