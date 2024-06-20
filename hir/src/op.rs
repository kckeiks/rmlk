use crate::attributes;
use crate::attributes::ValueInfo;
use crate::error::Error;
use crate::model::StringStringEntryProto;
use crate::node::Node;
use crate::onnx::{FunctionProto, OperatorSetIdProto};
use std::borrow::Cow;

pub struct OperatorSetId {
    domain: Option<String>,
    version: i64,
}

impl TryFrom<OperatorSetIdProto<'_>> for OperatorSetId {
    type Error = Error;

    fn try_from(value: OperatorSetIdProto) -> Result<Self, Self::Error> {
        Ok(Self {
            domain: value.domain.map(|cow| cow.to_string()),
            version: value.version.ok_or(Error::MissingField {
                name: "OperatorSetId::version".to_string(),
            })?,
        })
    }
}

pub struct Function {
    name: Option<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attribute: Attribute,
    node: Vec<Node>,
    doc_string: Option<String>,
    opset_import: Vec<OperatorSetId>,
    domain: Option<String>,
    overload: Option<String>,
    value_info: Vec<ValueInfo>,
    metadata_props: Vec<StringStringEntryProto>,
}

impl TryFrom<FunctionProto<'_>> for Function {
    type Error = Error;

    fn try_from(value: FunctionProto) -> Result<Self, Self::Error> {
        let attribute = if !value.attribute.is_empty() && !value.attribute_proto.is_empty() {
            return Err(Error::InvalidValue {
                field: "Function::attribute&Function::attribute_proto".to_string(),
                value: "cannot include both".to_string(),
            });
        } else if !value.attribute.is_empty() {
            Attribute::String {
                value: value
                    .attribute
                    .into_iter()
                    .map(|attr| attr.to_string())
                    .collect(),
            }
        } else {
            let mut attributes = Vec::new();
            for attr in value.attribute_proto {
                attributes.push(attr.try_into()?);
            }

            Attribute::Object { value: attributes }
        };

        let mut node = Vec::new();
        for n in value.node {
            node.push(n.try_into()?);
        }

        let mut opset_import = Vec::new();
        for op_set in value.opset_import {
            opset_import.push(op_set.try_into()?);
        }

        let mut value_info = Vec::new();
        for info in value.value_info {
            value_info.push(info.try_into()?);
        }

        let mut metadata_props = Vec::new();
        for props in value.metadata_props {
            metadata_props.push(props.into());
        }

        Ok(Self {
            name: value.name.map(|name| name.to_string()),
            inputs: value
                .input
                .into_iter()
                .map(|input| input.to_string())
                .collect(),
            outputs: value
                .output
                .into_iter()
                .map(|output| output.to_string())
                .collect(),
            attribute,
            node,
            doc_string: value.doc_string.map(|doc| doc.to_string()),
            opset_import,
            domain: value.domain.map(|domain| domain.to_string()),
            overload: value.overload.map(|overload| overload.to_string()),
            value_info,
            metadata_props,
        })
    }
}

enum Attribute {
    String { value: Vec<String> },
    Object { value: Vec<attributes::Attribute> },
}
