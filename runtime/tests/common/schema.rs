use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct GraphDef {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nodes: Vec<NodeDef>,
    pub tensors: Vec<()>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NodeDef {
    pub input: Option<Vec<String>>,
    pub output: Option<Vec<String>>,
    pub info: NodeTypeInfo,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum NodeTypeInfo {
    Op {
        name: String,
    },
    Value {
        name: String,
        shape: Option<Vec<usize>>,
    },
}
