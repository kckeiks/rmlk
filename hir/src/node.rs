use crate::attributes::Attribute;
use crate::model::StringStringEntryProto;

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
