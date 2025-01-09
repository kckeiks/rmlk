use std::fs;
use crate::builder::schema::GraphDef;

mod schema;

pub fn build() -> GraphDef {
    let reader = fs::read("./runtime/tests/definitions/add_simple.json").unwrap();
    let graph = serde_json::from_slice(reader.as_slice()).unwrap();

    graph
}