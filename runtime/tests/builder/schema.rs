use std::process::Output;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct GraphDef {
    inputs: Vec<String>,
    outputs: Vec<String>,
    nodes: Vec<NodeDef>,
    tensors: Vec<()>,
}

#[derive(Deserialize, Serialize)]
pub struct NodeDef {
    input: Option<Vec<String>>,
    output: Option<Vec<String>>,
    value: ValueDef,
}

#[derive(Deserialize, Serialize)]
pub enum ValueDef {
    Op(OpDef),
    Variable,
}

#[derive(Deserialize, Serialize)]
pub enum OpDef {
    Add,
}