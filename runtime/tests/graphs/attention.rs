use crate::{assert_close, load_f32};
use rmlk_runtime::{Builder, Value};
use rmlk_schema::{AttributeType, DataType, GraphBuilder, Op, Tensor};
use std::collections::HashMap;

/// Single-head attention fragment: MatMul, Transpose, Div, Trilu, Softmax, MatMul.
#[test]
fn attention_head() {
    let q = load_f32("attention_q.f32");
    let k = load_f32("attention_k.f32");
    let v = load_f32("attention_v.f32");
    let expected = load_f32("attention_out.f32");

    let mut g = GraphBuilder::new();
    g.input("q", DataType::Float, [1, 2, 4]).unwrap();
    g.input("k", DataType::Float, [1, 2, 4]).unwrap();
    g.input("v", DataType::Float, [1, 2, 4]).unwrap();
    g.constant("scale", Tensor::from_vec([], vec![2.0f32]).unwrap())
        .unwrap();

    g.value("kt", DataType::Float, [1, 4, 2]).unwrap();
    g.value("scores", DataType::Float, [1, 2, 2]).unwrap();
    g.value("scaled", DataType::Float, [1, 2, 2]).unwrap();
    g.value("masked", DataType::Float, [1, 2, 2]).unwrap();
    g.value("probs", DataType::Float, [1, 2, 2]).unwrap();
    g.value("out", DataType::Float, [1, 2, 4]).unwrap();

    g.op(Op::Transpose, &["k"], &["kt"])
        .unwrap()
        .attr("perm", AttributeType::Ints(vec![0, 2, 1]));
    g.op(Op::MatMul, &["q", "kt"], &["scores"]).unwrap();
    g.op(Op::Div, &["scores", "scale"], &["scaled"]).unwrap();
    g.op(Op::Trilu, &["scaled"], &["masked"])
        .unwrap()
        .attr("upper", AttributeType::Int(0));
    g.op(Op::Softmax, &["masked"], &["probs"])
        .unwrap()
        .attr("axis", AttributeType::Int(-1));
    g.op(Op::MatMul, &["probs", "v"], &["out"]).unwrap();
    g.output("out").unwrap();

    let graph = g.build().unwrap();
    let mut instance = Builder::from_graph(graph).unwrap().build().unwrap();

    let inputs: HashMap<String, Value> = [
        ("q".into(), (q, vec![1, 2, 4]).into()),
        ("k".into(), (k, vec![1, 2, 4]).into()),
        ("v".into(), (v, vec![1, 2, 4]).into()),
    ]
    .into();

    let mut outputs = instance.run(inputs).unwrap();
    let out: Vec<f32> = outputs.remove("out").unwrap().try_into().unwrap();
    assert_close(&out, &expected, 1e-5);
}
