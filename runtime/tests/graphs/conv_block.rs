use crate::{assert_close, load_f32};
use rmlk_runtime::{Builder, Value};
use rmlk_schema::{AttributeType, DataType, GraphBuilder, Op, Tensor};
use std::collections::HashMap;

/// Runs Conv, Relu, MaxPool, GlobalAveragePool, Reshape, and Gemm in sequence.
#[test]
fn conv_block() {
    let x = load_f32("conv_block_x.f32");
    let w = load_f32("conv_block_w.f32");
    let fc = load_f32("conv_block_fc.f32");
    let expected = load_f32("conv_block_out.f32");

    let mut g = GraphBuilder::new();
    g.input("x", DataType::Float, [1, 1, 4, 4]).unwrap();
    g.constant("w", Tensor::from_vec([2, 1, 3, 3], w).unwrap())
        .unwrap();
    g.constant("fc", Tensor::from_vec([2, 3], fc).unwrap())
        .unwrap();
    g.constant("flat_shape", Tensor::from_vec([2], vec![1i64, 2]).unwrap())
        .unwrap();

    g.value("conv_y", DataType::Float, [1, 2, 2, 2]).unwrap();
    g.value("relu_y", DataType::Float, [1, 2, 2, 2]).unwrap();
    g.value("pool_y", DataType::Float, [1, 2, 1, 1]).unwrap();
    g.value("gap_y", DataType::Float, [1, 2, 1, 1]).unwrap();
    g.value("flat", DataType::Float, [1, 2]).unwrap();
    g.value("out", DataType::Float, [1, 3]).unwrap();

    g.op(Op::Conv, &["x", "w"], &["conv_y"])
        .unwrap()
        .attr("pads", AttributeType::Ints(vec![0, 0, 0, 0]))
        .attr("strides", AttributeType::Ints(vec![1, 1]))
        .attr("dilations", AttributeType::Ints(vec![1, 1]));
    g.op(Op::Relu, &["conv_y"], &["relu_y"]).unwrap();
    g.op(Op::MaxPool, &["relu_y"], &["pool_y"])
        .unwrap()
        .attr("kernel_shape", AttributeType::Ints(vec![2, 2]))
        .attr("strides", AttributeType::Ints(vec![2, 2]))
        .attr("pads", AttributeType::Ints(vec![0, 0, 0, 0]));
    g.op(Op::GlobalAveragePool, &["pool_y"], &["gap_y"])
        .unwrap();
    g.op(Op::Reshape, &["gap_y", "flat_shape"], &["flat"])
        .unwrap();
    g.op(Op::Gemm, &["flat", "fc"], &["out"]).unwrap();
    g.output("out").unwrap();

    let graph = g.build().unwrap();
    let mut instance = Builder::from_graph(graph).unwrap().build().unwrap();

    let inputs: HashMap<String, Value> = [("x".into(), (x, vec![1, 1, 4, 4]).into())].into();
    let mut outputs = instance.run(inputs).unwrap();
    let out: Vec<f32> = outputs.remove("out").unwrap().try_into().unwrap();
    assert_close(&out, &expected, 1e-5);
}
