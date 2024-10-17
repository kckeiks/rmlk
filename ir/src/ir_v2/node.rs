use crate::attributes::Attribute;
use crate::error::Error;
use crate::model::StringStringEntryProto;
use crate::onnx::NodeProto;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Deserialize, Serialize)]
pub struct Node {
    // Input nodes.
    pub input: Vec<u64>,
    // Output nodes.
    pub output: Vec<u64>,
    // An identifier for this node in a graph.
    pub id: u64,
    // The symbolic identifier of the Operator to execute.
    pub op_type: Option<u32>,
    // Additional named attributes.
    pub attribute: Vec<Attribute>,
    // Optional name of node.
    pub name: String,
}
