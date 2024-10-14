use crate::attributes::Attribute;
use crate::error::Error;
use crate::model::StringStringEntryProto;
use crate::onnx::NodeProto;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Deserialize, Serialize)]
pub struct Node {
    // Input nodes.
    pub input: Vec<String>,
    // Output nodes.
    pub output: Vec<String>,
    // An optional identifier for this node in a graph.
    // This field MAY be absent in this version of the IR.
    pub name: Option<String>,
    // The symbolic identifier of the Operator to execute.
    pub op_type: Option<String>,
    // The domain of the OperatorSet that specifies the operator named by op_type.
    pub domain: Option<String>,
    // Overload identifier, used only to map this to a model-local function.
    pub overload: Option<String>,
    // Additional named attributes.
    pub attribute: Vec<Attribute>,
    // A human-readable documentation for this node. Markdown is allowed.
    pub doc_string: Option<String>,
    // Named metadata values; keys should be distinct.
    pub metadata_props: Vec<StringStringEntryProto>,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            output: self.output.clone(),
            name: self.name.clone(),
            op_type: self.op_type.clone(),
            domain: self.domain.clone(),
            // Todo: Finish.
            overload: None,
            attribute: vec![],
            doc_string: None,
            metadata_props: vec![],
        }
    }
}

impl TryFrom<NodeProto<'_>> for Node {
    type Error = Error;

    fn try_from(value: NodeProto) -> Result<Self, Self::Error> {
        let mut attribute = Vec::new();
        for attr in value.attribute {
            attribute.push(attr.try_into()?);
        }

        let mut metadata_props = Vec::new();
        for props in value.metadata_props {
            metadata_props.push(props.into());
        }

        Ok(Self {
            input: value
                .input
                .into_iter()
                .map(|input| input.to_string())
                .collect(),
            output: value
                .output
                .into_iter()
                .map(|output| output.to_string())
                .collect(),
            name: value.name.map(|name| name.to_string()),
            op_type: value.op_type.map(|op_type| op_type.to_string()),
            domain: value.domain.map(|domain| domain.to_string()),
            overload: value.overload.map(|overload| overload.to_string()),
            attribute,
            doc_string: value.doc_string.map(|doc| doc.to_string()),
            metadata_props,
        })
    }
}
