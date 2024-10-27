use crate::error::Error;
use crate::onnx::ty_proto::OneOfvalue;
use crate::onnx::{tensor_shape_proto, TypeProto};
use crate::{onnx, DataType};
use serde::{Deserialize, Serialize};

/// Defines information on value, including the name, the type, and
/// the shape of the value.
#[derive(Debug, Deserialize, Serialize)]
pub struct ValueInfo {
    /// This field MUST be present in this version of the IR.
    pub id: usize,
    /// This field MUST be present in this version of the IR for
    /// inputs and outputs of the top-level graph.
    pub ty: Option<TypeValue>,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum TypeValue {
    Tensor { ty: i32, dims: Vec<usize> },
}

impl TypeValue {
    pub fn from_type_proto(type_proto: TypeProto) -> Result<Option<Self>, Error> {
        let ty = match type_proto.value {
            OneOfvalue::tensor_type(tensor) => {
                if tensor.elem_type == Some(onnx::tensor_proto::DataType::UNDEFINED as i32) {
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

                let mut dims = Vec::new();
                for s in &tensor.shape.ok_or(Error::Unknown)?.dim {
                    if let tensor_shape_proto::mod_Dimension::OneOfvalue::dim_value(v) = s.value {
                        dims.push(usize::try_from(v).map_err(|_| Error::InvalidValue {
                            field: "Dimension::value".to_string(),
                            value: v.to_string(),
                        })?);
                    } else {
                        // Todo: add support for other types here.
                        return Err(Error::NotSupportedD);
                    }
                }

                Some(Self::Tensor {
                    ty: tensor.elem_type.unwrap(),
                    dims,
                })
            }
            OneOfvalue::None => return Ok(None),
            _ => unimplemented!(),
        };

        Ok(ty)
    }

    pub fn ty(&self) -> i32 {
        match self {
            TypeValue::Tensor { ty, .. } => *ty,
        }
    }

    pub fn dims(&self) -> &Vec<usize> {
        match self {
            TypeValue::Tensor { dims, .. } => &dims,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Value {
    pub dtype: DataType,
    pub dims: Vec<usize>,
    pub name: String,
}

impl Value {
    pub fn new(dtype: DataType, dims: Vec<usize>, name: String) -> Self {
        Self { name, dtype, dims }
    }
}
