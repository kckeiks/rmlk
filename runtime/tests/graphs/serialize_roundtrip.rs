use crate::assert_close;
use rmlk_runtime::{Builder, Value};
use rmlk_schema::{DataType, Graph, GraphBuilder, Op};
use std::collections::HashMap;

fn add_graph() -> Graph {
    let mut g = GraphBuilder::new();
    g.input("a", DataType::Float, [2, 2]).unwrap();
    g.input("b", DataType::Float, [2, 2]).unwrap();
    g.value("y", DataType::Float, [2, 2]).unwrap();
    g.op(Op::Add, &["a", "b"], &["y"]).unwrap();
    g.output("y").unwrap();
    g.build().unwrap()
}

/// Serializing a graph with bincode and loading it through
/// [`Builder::with_model_from_memory`] must produce the same run results as
/// building directly with [`Builder::from_graph`].
#[test]
fn bincode_round_trip_matches_from_graph() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let inputs = || -> HashMap<String, Value> {
        [
            ("a".into(), (a.clone(), vec![2, 2]).into()),
            ("b".into(), (b.clone(), vec![2, 2]).into()),
        ]
        .into()
    };

    let mut from_graph = Builder::from_graph(add_graph()).unwrap().build().unwrap();
    let mut out_direct = from_graph.run(inputs()).unwrap();
    let direct: Vec<f32> = out_direct.remove("y").unwrap().try_into().unwrap();

    let bytes = bincode::serialize(&add_graph()).expect("serialize graph");
    let mut from_memory = Builder::with_model_from_memory(bytes.into_boxed_slice())
        .unwrap()
        .build()
        .unwrap();
    let mut out_memory = from_memory.run(inputs()).unwrap();
    let via_memory: Vec<f32> = out_memory.remove("y").unwrap().try_into().unwrap();

    assert_close(&via_memory, &direct, 0.0);
    assert_eq!(direct, vec![11.0, 22.0, 33.0, 44.0]);
}
