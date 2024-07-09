use crate::error::Error;
use crate::graph::Graph;
use crate::onnx;
use crate::onnx::ModelProto;
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

impl TryFrom<ModelProto<'_>> for Model {
    type Error = Error;

    fn try_from(value: ModelProto) -> Result<Self, Self::Error> {
        let mut metadata_props = Vec::new();
        for metadata in value.metadata_props {
            metadata_props.push(metadata.into());
        }

        let mut functions = Vec::new();
        for f in value.functions {
            functions.push(f.try_into()?);
        }

        let mut opset_import = Vec::new();
        for opset in value.opset_import {
            opset_import.push(opset.try_into()?);
        }

        Ok(Self {
            ir_version: value
                .ir_version
                .ok_or(Error::MissingField {
                    name: "Model::ir_version".to_string(),
                })?
                .try_into()?,
            opset_import,
            producer_name: value.producer_name.map(|str| str.to_string()),
            producer_version: value.producer_version.map(|str| str.to_string()),
            domain: value.domain.map(|str| str.to_string()),
            model_version: value.model_version,
            doc_string: value.doc_string.map(|str| str.to_string()),
            graph: value.graph.map(|g| g.try_into()).transpose()?,
            metadata_props,
            functions,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StringStringEntryProto {
    key: Option<String>,
    value: Option<String>,
}

impl From<onnx::StringStringEntryProto<'_>> for StringStringEntryProto {
    fn from(value: onnx::StringStringEntryProto) -> Self {
        Self {
            key: value.key.map(|k| k.to_string()),
            value: value.value.map(|v| v.to_string()),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TensorAnnotation {
    tensor_name: Option<String>,
    quant_parameter_tensor_names: Vec<StringStringEntryProto>,
}

impl From<onnx::TensorAnnotation<'_>> for TensorAnnotation {
    fn from(value: onnx::TensorAnnotation) -> Self {
        Self {
            tensor_name: value.tensor_name.map(|name| name.to_string()),
            quant_parameter_tensor_names: value
                .quant_parameter_tensor_names
                .into_iter()
                .map(From::from)
                .collect(),
        }
    }
}
