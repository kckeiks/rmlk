use crate::error::Error;
use crate::graph::Graph;
use crate::op::{Function, OperatorSetId};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub enum Version {
    Ir2024 = 1,
}

impl TryFrom<i64> for Version {
    type Error = Error;

    fn try_from(_value: i64) -> Result<Self, Self::Error> {
        // Todo: come back this.
        Ok(Self::Ir2024)
    }
}

/// Models
///
/// ModelProto is a top-level file/container format for bundling a ML model and
/// associating its computation graph with metadata.
///
/// The semantics of the model are described by the associated GraphProto's.
#[derive(Deserialize, Serialize)]
pub struct Model {
    /// The version of the IR this model targets.
    pub ir_version: Version,
    /// The OperatorSets this model relies on.
    /// All ModelProtos MUST have at least one entry that
    /// specifies which version of the RMLK OperatorSet is
    /// being imported.
    ///
    /// All nodes in the ModelProto's graph will bind against the operator
    /// with the same-domain/same-op_type operator with the HIGHEST version
    /// in the referenced operator sets.
    pub opset_import: Vec<OperatorSetId>,
    /// The name of the framework or tool used to generate this model.
    /// This field SHOULD be present to indicate which implementation/tool/framework
    /// emitted the model.
    pub producer_name: Option<String>,
    /// The version of the framework or tool used to generate this model.
    /// This field SHOULD be present to indicate which implementation/tool/framework
    /// emitted the model.
    pub producer_version: Option<String>,
    /// Domain name of the model.
    /// We use reverse domain names as name space indicators. For example:
    /// `com.facebook.fair` or `com.microsoft.cognitiveservices`
    ///
    /// Together with `model_version` and GraphProto.name, this forms the unique identity of
    /// the graph.
    pub domain: Option<String>,
    /// The version of the graph encoded. See Version enum below.
    pub model_version: Option<i64>,
    /// A human-readable documentation for this model. Markdown is allowed.
    pub doc_string: Option<String>,
    /// The parameterized graph that is evaluated to execute the model.
    pub graph: Option<Graph>,
    /// Named metadata values; keys should be distinct.
    pub metadata_props: Vec<StringStringEntryProto>,
    /// A list of function protos local to the model.
    ///
    /// The (domain, name, overload) tuple must be unique across the function protos in this list.
    /// In case of any conflicts the behavior (whether the model local functions are given higher priority,
    /// or standard operator sets are given higher priotity or this is treated as error) is defined by
    /// the runtimes.
    ///
    /// The operator sets imported by FunctionProto should be compatible with the ones
    /// imported by ModelProto and other model local FunctionProtos.
    /// Example, if same operator set say 'A' is imported by a FunctionProto and ModelProto
    /// or by 2 FunctionProtos then versions for the operator set may be different but,
    /// the operator schema returned for op_type, domain, version combination
    /// for both the versions should be same for every node in the function body.
    ///
    /// One FunctionProto can reference other FunctionProto in the model, however, recursive reference
    /// is not allowed.
    pub functions: Vec<Function>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StringStringEntryProto {
    key: Option<String>,
    value: Option<String>,
}
