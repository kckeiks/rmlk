use quick_protobuf::{BytesReader, MessageRead};
use rmlk_hir::{Model, ModelProto};
use std::fs;

fn main() {
    let model = fs::read("/Users/acadia/Repo/Llama-2-7b-ONNX/FP32-Chat/LlamaV2_7B_FT_float32.onnx")
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

    let res = rmlk_runtime::parse::parse_ir_graph(rmlk_model.graph.unwrap()).unwrap();

    println!("Nodes {:?}", res.nodes().count());
    println!("Outputs {:?}", res.outputs().count());
}
