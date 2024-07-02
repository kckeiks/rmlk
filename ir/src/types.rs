use crate::error::Error;
use crate::onnx::mod_TypeProto::OneOfvalue;
use crate::onnx::TypeProto;
use crate::tensor::TensorShape;
use crate::{onnx, DataType, DimensionValue};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Type {
    pub value: Option<TypeValue>,
    // An optional denotation can be used to denote the whole
    // type with a standard semantic description as to what is
    // stored inside. Refer to https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition
    // for pre-defined type denotations.
    pub denotation: Option<String>,
}

impl Type {
    // Todo: Should we support other types such as maps, sparse tensors, etc.?
    pub fn get_tensor_info(&self) -> Option<(DataType, Option<Vec<usize>>)> {
        match self.value.as_ref()? {
            TypeValue::Tensor { elem_type, shape } => {
                let mut dims = Vec::new();
                for s in &shape.dim {
                    if let DimensionValue::Value(v) = s
                        .value
                        .as_ref()
                        .ok_or(Error::MissingField {
                            name: "Dimension::value".to_string(),
                        })
                        .ok()?
                    {
                        dims.push(
                            usize::try_from(*v)
                                .map_err(|_| Error::InvalidValue {
                                    field: "Dimension::value".to_string(),
                                    value: v.to_string(),
                                })
                                .ok()?,
                        );
                    } else {
                        // Todo: For now let's abort creating a shape when it includes an
                        // unknown dimension and let the graph infer the shape from the inputs.
                        // It is not clear at the moment if we need to keep this around.
                        return Some((DataType::try_from(*elem_type).ok().unwrap(), None));
                    }
                }

                if dims.is_empty() {
                    Some((DataType::try_from(*elem_type).ok().unwrap(), None))
                } else {
                    Some((DataType::try_from(*elem_type).ok().unwrap(), Some(dims)))
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub enum TypeValue {
    Map {
        /// This field MUST have a valid TensorProto.DataType value.
        /// This field MUST be present for this version of the IR.
        /// This field MUST refer to an integral type ([U]INT{8|16|32|64}) or STRING.
        key: i32,
        /// This field MUST be present for this version of the IR.
        value: Box<Type>,
    },
    Tensor {
        /// This field MUST NOT have the value of UNDEFINED.
        /// This field MUST have a valid TensorProto.DataType value.
        /// This field MUST be present for this version of the IR.
        elem_type: i32,
        shape: TensorShape,
    },
    /// The type and optional shape of each element of the sequence.
    /// This field MUST be present for this version of the IR.
    Sequence { elem_type: Box<Type> },
    SparseTensor {
        /// This field MUST NOT have the value of UNDEFINED
        /// This field MUST have a valid TensorProto.DataType value.
        /// This field MUST be present for this version of the IR.
        elem_type: i32,
        shape: TensorShape,
    },
    Optional {
        // The type and optional shape of the element wrapped.
        // This field MUST be present for this version of the IR.
        // Possible values correspond to OptionalProto.DataType enum.
        elem_type: Box<Type>,
    },
}

impl TryFrom<TypeProto<'_>> for Type {
    type Error = Error;

    fn try_from(value: TypeProto) -> Result<Self, Self::Error> {
        let ty = match value.value {
            OneOfvalue::tensor_type(tensor) => {
                if tensor.elem_type == Some(onnx::mod_TensorProto::DataType::UNDEFINED as i32) {
                    return Err(Error::InvalidValue {
                        field: "Type::value".to_string(),
                        value: "UNDEFINED is not valid".to_string(),
                    });
                }

                if tensor.elem_type.is_none() {
                    return Err(Error::MissingField {
                        name: "Type::value::Tensor::elem_type".to_string(),
                    });
                }

                if tensor.shape.is_none() {
                    return Err(Error::MissingField {
                        name: "Type::value::Tensor::shape".to_string(),
                    });
                }

                Some(TypeValue::Tensor {
                    elem_type: tensor.elem_type.unwrap(),
                    shape: tensor.shape.unwrap().try_into()?,
                })
            }
            OneOfvalue::sequence_type(seq) => {
                let type_proto = seq.elem_type.ok_or(Error::MissingField {
                    name: "Type::value::Sequence::elem_type".to_string(),
                })?;
                Some(TypeValue::Sequence {
                    elem_type: Box::new((*type_proto).try_into()?),
                })
            }
            OneOfvalue::map_type(map) => {
                let type_proto = map.value_type.ok_or(Error::MissingField {
                    name: "Type::value::Map::value_type".to_string(),
                })?;
                Some(TypeValue::Map {
                    key: map.key_type.ok_or(Error::MissingField {
                        name: "Type::value::Map::key_type".to_string(),
                    })?,
                    value: Box::new((*type_proto).try_into()?),
                })
            }
            OneOfvalue::optional_type(optional) => {
                let type_proto = optional.elem_type.ok_or(Error::MissingField {
                    name: "Type::value::Map::key_type".to_string(),
                })?;
                Some(TypeValue::Optional {
                    elem_type: Box::new((*type_proto).try_into()?),
                })
            }
            OneOfvalue::sparse_tensor_type(sparse_tensor) => {
                if sparse_tensor.elem_type
                    == Some(onnx::mod_TensorProto::DataType::UNDEFINED as i32)
                {
                    return Err(Error::InvalidValue {
                        field: "Type::value".to_string(),
                        value: "UNDEFINED is not valid".to_string(),
                    });
                }

                if sparse_tensor.elem_type.is_none() {
                    return Err(Error::MissingField {
                        name: "Type::value::Tensor::elem_type".to_string(),
                    });
                }

                if sparse_tensor.shape.is_none() {
                    return Err(Error::MissingField {
                        name: "Type::value::Tensor::shape".to_string(),
                    });
                }
                Some(TypeValue::SparseTensor {
                    elem_type: sparse_tensor.elem_type.unwrap(),
                    shape: sparse_tensor.shape.unwrap().try_into()?,
                })
            }
            OneOfvalue::None => None,
        };

        Ok(Self {
            value: ty,
            denotation: value.denotation.map(|denotation| denotation.to_string()),
        })
    }
}
