use image::GenericImageView;
use ndarray::Array;
use quick_protobuf::{BytesReader, MessageRead};
use rmlk_ir::{Model, ModelProto};
use rmlk_runtime::Builder;
use std::fs;
use std::path::Path;

fn main() {
    let mut args = std::env::args();
    args.next();

    let Some(path) = args.next() else {
        println!("missing image path argument");
        std::process::exit(1);
    };

    // Open image.
    let original_img = image::open(Path::new(&path)).unwrap();

    // Preprocessing.
    let img = original_img.thumbnail(224, 224);
    let mut input = Array::zeros((1, 3, 224, 224));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let model = fs::read("/home/mmeier/Downloads/resnet34.onnx").expect("bad");
    let mut reader = BytesReader::from_bytes(&model);
    let model_proto = ModelProto::from_reader(&mut reader, &model).unwrap();
    // println!(
    //     "{:?}",
    //     &model_proto.graph.as_ref().unwrap().initializer[0]
    //         .raw_data
    //         .as_ref()
    //         .unwrap()
    //         .len()
    // );
    let rmlk_model: Model = model_proto.try_into().unwrap();
    let graph = rmlk_runtime::parse::parse_ir_graph(rmlk_model.graph.unwrap()).unwrap();

    // for (id, node) in graph.nodes().enumerate() {
    //     println!("id={id}, node={:?}", node.op());
    // }

    let builder = Builder::new(graph);
    let mut session = builder.build().unwrap();
    let output = session.run(input.into_raw_vec()).unwrap();
    println!("output: {:?}", output[0])
}
