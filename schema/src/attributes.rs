use crate::error::Error;
use crate::onnx::AttributeProto;
use crate::{onnx, DataType, Tensor};
use serde::{Deserialize, Serialize};

/// Attributes
///
/// A named attribute containing either singular float, integer, string, graph,
/// and tensor values, or repeated float, integer, string, graph, and tensor values.
/// An AttributeProto MUST contain the name field, and *only one* of the
/// following content fields, effectively enforcing a C/C++ union equivalent.
#[derive(Debug, Deserialize, Serialize)]
pub struct Attribute {
    /// The name of the attribute.
    pub name: String,
    /// If ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
    /// In this case, this AttributeProto does not contain data, and it's a reference of attribute
    /// in parent scope.
    /// NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
    pub ref_attr_name: Option<String>,
    /// The type of the attribute.
    pub ty: AttributeType,
    /// A human-readable documentation for this attribute. Markdown is allowed.
    pub doc_string: Option<String>,
}

impl Attribute {
    pub fn ints(&self) -> Option<&[i32]> {
        match &self.ty {
            AttributeType::Ints(value) => Some(value),
            _ => None,
        }
    }

    pub fn float(&self) -> Option<f32> {
        match &self.ty {
            AttributeType::Float(value) => Some(*value),
            _ => None,
        }
    }

    pub fn floats(&self) -> Option<&[f32]> {
        match &self.ty {
            AttributeType::Floats(value) => Some(value),
            _ => None,
        }
    }

    pub fn int(&self) -> Option<i32> {
        match &self.ty {
            AttributeType::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn dtype(&self) -> Option<DataType> {
        match self.ty {
            AttributeType::DataType(dtype) => Some(dtype),
            _ => None,
        }
    }

    pub fn tensor(&self) -> Option<&Tensor> {
        match &self.ty {
            AttributeType::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn string(&self) -> Option<&Vec<u8>> {
        match &self.ty {
            AttributeType::String(s) => Some(s),
            _ => None,
        }
    }
}

impl TryFrom<AttributeProto<'_>> for Attribute {
    type Error = Error;

    fn try_from(value: AttributeProto) -> Result<Self, Self::Error> {
        let attribute_ty = match value.type_pb.ok_or(Error::MissingField {
            name: "Attribute::type".to_string(),
        })? {
            onnx::attributte_proto::AttributeType::UNDEFINED => {
                return Err(Error::InvalidValue {
                    field: "Attribute::type".to_string(),
                    value: "undefined".to_string(),
                });
            }
            onnx::attributte_proto::AttributeType::FLOAT => {
                AttributeType::Float(value.f.ok_or(Error::MissingField {
                    name: "Attribute::f".to_string(),
                })?)
            }
            // Todo: Address casting.
            onnx::attributte_proto::AttributeType::INT => {
                AttributeType::Int(value.i.ok_or(Error::MissingField {
                    name: "Attribute::i".to_string(),
                })? as i32)
            }
            onnx::attributte_proto::AttributeType::STRING => AttributeType::String(
                value
                    .s
                    .ok_or(Error::MissingField {
                        name: "Attribute::s".to_string(),
                    })?
                    .to_vec(),
            ),
            onnx::attributte_proto::AttributeType::TENSOR => AttributeType::Tensor(
                crate::Tensor::from_onnx_tensor(value.t.ok_or(Error::MissingField {
                    name: "Attribute::t".to_string(),
                })?)?,
            ),
            onnx::attributte_proto::AttributeType::FLOATS => AttributeType::Floats(value.floats),
            // Todo: Address casting.
            onnx::attributte_proto::AttributeType::INTS => {
                AttributeType::Ints(value.ints.iter().map(|num| *num as i32).collect())
            }
            onnx::attributte_proto::AttributeType::STRINGS => {
                AttributeType::Strings(value.strings.into_iter().map(|s| s.to_vec()).collect())
            }
            onnx::attributte_proto::AttributeType::TENSORS => {
                let mut tensors = Vec::new();
                for tensor_proto in value.tensors.into_iter() {
                    tensors.push(crate::Tensor::from_onnx_tensor(tensor_proto)?);
                }
                AttributeType::Tensors(tensors)
            }
            ty => panic!("unsupported attribute type {ty:?}"),
        };

        Ok(Self {
            name: value
                .name
                .map(|name| name.to_string())
                .ok_or(Error::MissingField {
                    name: "Attribute::name".to_string(),
                })?,
            ref_attr_name: value.ref_attr_name.map(|name| name.to_string()),
            doc_string: value.doc_string.map(|doc| doc.to_string()),
            ty: attribute_ty,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub enum AttributeType {
    Float(f32),
    Int(i32),
    String(Vec<u8>),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Ints(Vec<i32>),
    Strings(Vec<Vec<u8>>),
    Tensors(Vec<Tensor>),
    DataType(DataType),
}
