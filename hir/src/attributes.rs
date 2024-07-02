use serde::{Deserialize, Serialize};
use crate::error::Error;
use crate::graph::Graph;
use crate::model::StringStringEntryProto;
use crate::onnx::{self, AttributeProto, ValueInfoProto};
use crate::tensor::{SparseTensor, Tensor};
use crate::types::Type;

/// Attributes
///
/// A named attribute containing either singular float, integer, string, graph,
/// and tensor values, or repeated float, integer, string, graph, and tensor values.
/// An AttributeProto MUST contain the name field, and *only one* of the
/// following content fields, effectively enforcing a C/C++ union equivalent.
#[derive(Deserialize, Serialize)]
pub struct Attribute {
    /// The name of the attribute.
    name: String,
    /// If ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
    /// In this case, this AttributeProto does not contain data, and it's a reference of attribute
    /// in parent scope.
    /// NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
    ref_attr_name: Option<String>,
    /// A human-readable documentation for this attribute. Markdown is allowed.
    doc_string: Option<String>,
    /// The type of the attribute.
    ty: AttributeType,
}

impl TryFrom<AttributeProto<'_>> for Attribute {
    type Error = Error;

    fn try_from(value: AttributeProto) -> Result<Self, Self::Error> {
        let attribute_ty = match value.type_pb.ok_or(Error::MissingField {
            name: "Attribute::type".to_string(),
        })? {
            onnx::mod_AttributeProto::AttributeType::UNDEFINED => {
                return Err(Error::InvalidValue {
                    field: "Attribute::type".to_string(),
                    value: "undefined".to_string(),
                });
            }
            onnx::mod_AttributeProto::AttributeType::FLOAT => {
                AttributeType::Float(value.f.ok_or(Error::MissingField {
                    name: "Attribute::f".to_string(),
                })?)
            }
            onnx::mod_AttributeProto::AttributeType::INT => {
                AttributeType::Int(value.i.ok_or(Error::MissingField {
                    name: "Attribute::i".to_string(),
                })?)
            }
            onnx::mod_AttributeProto::AttributeType::STRING => AttributeType::String(
                value
                    .s
                    .ok_or(Error::MissingField {
                        name: "Attribute::s".to_string(),
                    })?
                    .to_vec(),
            ),
            onnx::mod_AttributeProto::AttributeType::TENSOR => AttributeType::Tensor(
                Tensor::from_onnx_tensor(value
                    .t
                    .ok_or(Error::MissingField {
                        name: "Attribute::t".to_string(),
                    })?)?,
            ),
            onnx::mod_AttributeProto::AttributeType::GRAPH => AttributeType::Graph(
                value
                    .g
                    .ok_or(Error::MissingField {
                        name: "Attribute::g".to_string(),
                    })?
                    .try_into()?,
            ),
            onnx::mod_AttributeProto::AttributeType::SPARSE_TENSOR => AttributeType::SparseTensor(
                value
                    .sparse_tensor
                    .ok_or(Error::MissingField {
                        name: "Attribute::sparse_tensor".to_string(),
                    })?
                    .try_into()?,
            ),
            onnx::mod_AttributeProto::AttributeType::TYPE_PROTO => AttributeType::Type(
                value
                    .tp
                    .ok_or(Error::MissingField {
                        name: "Attribute::tp".to_string(),
                    })?
                    .try_into()?,
            ),
            onnx::mod_AttributeProto::AttributeType::FLOATS => AttributeType::Floats(value.floats),
            onnx::mod_AttributeProto::AttributeType::INTS => AttributeType::Ints(value.ints),
            onnx::mod_AttributeProto::AttributeType::STRINGS => {
                AttributeType::Strings(value.strings.into_iter().map(|s| s.to_vec()).collect())
            }
            onnx::mod_AttributeProto::AttributeType::TENSORS => {
                let mut tensors = Vec::new();
                for tensor_proto in value.tensors.into_iter() {
                    tensors.push(Tensor::from_onnx_tensor(tensor_proto)?);
                }
                AttributeType::Tensors(tensors)
            }
            onnx::mod_AttributeProto::AttributeType::GRAPHS => {
                let mut graphs = Vec::new();
                for graph_proto in value.graphs.into_iter() {
                    graphs.push(graph_proto.try_into()?);
                }
                AttributeType::Graphs(graphs)
            }
            onnx::mod_AttributeProto::AttributeType::SPARSE_TENSORS => {
                let mut tensors = Vec::new();
                for tensor_proto in value.sparse_tensors.into_iter() {
                    tensors.push(tensor_proto.try_into()?);
                }
                AttributeType::SparseTensors(tensors)
            }
            onnx::mod_AttributeProto::AttributeType::TYPE_PROTOS => {
                let mut types = Vec::new();
                for type_proto in value.type_protos.into_iter() {
                    types.push(type_proto.try_into()?);
                }
                AttributeType::Types(types)
            }
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

#[derive(Deserialize, Serialize)]
enum AttributeType {
    Float(f32),
    Int(i64),
    String(Vec<u8>),
    Tensor(Tensor),
    Graph(Graph),
    SparseTensor(SparseTensor),
    Type(Type),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    Ints(Vec<i64>),
    Strings(Vec<Vec<u8>>),
    Tensors(Vec<Tensor>),
    Graphs(Vec<Graph>),
    SparseTensors(Vec<SparseTensor>),
    Types(Vec<Type>),
}

/// Defines information on value, including the name, the type, and
/// the shape of the value.
#[derive(Debug, Deserialize, Serialize)]
pub struct ValueInfo {
    /// This field MUST be present in this version of the IR.
    pub name: String,
    /// This field MUST be present in this version of the IR for
    /// inputs and outputs of the top-level graph.
    pub ty: Option<Type>,
    /// A human-readable documentation for this value. Markdown is allowed.
    pub doc_string: Option<String>,
    /// Named metadata values; keys should be distinct.
    pub metadata_props: Vec<StringStringEntryProto>,
}

impl TryFrom<ValueInfoProto<'_>> for ValueInfo {
    type Error = Error;

    fn try_from(value: ValueInfoProto) -> Result<Self, Self::Error> {
        let mut metadata_props = Vec::new();
        for metadata in value.metadata_props {
            metadata_props.push(metadata.into());
        }

        Ok(Self {
            name: value
                .name
                .map(|name| name.to_string())
                .ok_or(Error::MissingField {
                    name: "ValueInfo::name".to_string(),
                })?,
            ty: value.type_pb.map(|ty| ty.try_into()).transpose()?,
            doc_string: value.doc_string.map(|doc| doc.to_string()),
            metadata_props,
        })
    }
}
