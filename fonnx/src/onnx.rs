use rmlk_ir::{dimension_proto, tensor_proto, ty_proto, NodeProto, TensorProto, ValueInfoProto};
use std::borrow::Cow;
use std::fmt::{Debug, Formatter};

#[derive(Debug)]
pub enum Category {
    Input,
    Output,
    Initializer,
    InnerNode,
}

pub struct NodeInfo<'a> {
    pub name: Option<Cow<'a, str>>,
    pub tensor: Option<TensorInfo>,
    pub input: Option<Vec<Cow<'a, str>>>,
    pub output: Option<Vec<Cow<'a, str>>>,
    pub op_type: Option<Cow<'a, str>>,
    pub domain: Option<Cow<'a, str>>,
    pub overload: Option<Cow<'a, str>>,
    // pub attribute: Vec<AttributeProto<'a>>,
    pub doc_string: Option<Cow<'a, str>>,
    // pub metadata_props: Vec<StringStringEntryProto<'a>>,
}

impl<'a> TryFrom<ValueInfoProto<'a>> for NodeInfo<'a> {
    type Error = anyhow::Error;
    fn try_from(value: ValueInfoProto<'a>) -> Result<Self, Self::Error> {
        let tensor_info = match value.type_pb {
            Some(type_pb) => {
                let info = match type_pb.value {
                    ty_proto::OneOfvalue::tensor_type(ty_proto::Tensor { elem_type, shape }) => {
                        let dtype = tensor_proto::DataType::try_from(elem_type.unwrap_or(0))?;
                        let mut dims = Vec::new();
                        if let Some(tensor_shape_proto) = shape {
                            for dimension in tensor_shape_proto.dim {
                                match dimension.value {
                                    dimension_proto::OneOfvalue::dim_value(val) => {
                                        dims.push(val);
                                    }
                                    dimension_proto::OneOfvalue::dim_param(_) => {}
                                    dimension_proto::OneOfvalue::None => {}
                                }
                            }
                        }
                        TensorInfo { dims, dtype }
                    }
                    ty_proto::OneOfvalue::sequence_type(_) => unimplemented!(),
                    ty_proto::OneOfvalue::map_type(_) => unimplemented!(),
                    ty_proto::OneOfvalue::optional_type(_) => unimplemented!(),
                    ty_proto::OneOfvalue::sparse_tensor_type(_) => unimplemented!(),
                    ty_proto::OneOfvalue::None => unimplemented!(),
                };

                Some(info)
            }
            _ => None,
        };

        Ok(Self {
            name: value.name,
            tensor: tensor_info,
            input: None,
            output: None,
            op_type: None,
            domain: None,
            overload: None,
            doc_string: None,
        })
    }
}

impl<'a> TryFrom<TensorProto<'a>> for NodeInfo<'a> {
    type Error = anyhow::Error;

    fn try_from(value: TensorProto<'a>) -> Result<Self, Self::Error> {
        let dtype = tensor_proto::DataType::try_from(value.data_type.unwrap_or(0))?;
        let tensor = TensorInfo {
            dtype,
            dims: value.dims,
        };

        Ok(NodeInfo {
            name: value.name,
            tensor: Some(tensor),
            input: None,
            output: None,
            op_type: None,
            domain: None,
            overload: None,
            doc_string: None,
        })
    }
}

impl Debug for NodeWithMetadata<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("NodeWithMetadata");
        debug_struct.field("category", &self.category);

        if let Some(name) = &self.node.name {
            debug_struct.field("name", name);
        } else {
            debug_struct.field("name", &"unknown");
        }

        if let Some(tensor_info) = &self.node.tensor {
            debug_struct.field("dtype", &tensor_info.dtype);
            debug_struct.field("dimensions", &tensor_info.dims);
        }

        if let Some(input) = &self.node.input {
            debug_struct.field("input", input);
        }

        if let Some(output) = &self.node.output {
            debug_struct.field("output", output);
        }

        debug_struct.finish()
    }
}

impl<'a> From<NodeProto<'a>> for NodeInfo<'a> {
    fn from(value: NodeProto<'a>) -> Self {
        NodeInfo {
            name: value.name,
            tensor: None,
            input: Some(value.input),
            output: Some(value.output),
            op_type: None,
            domain: None,
            overload: None,
            doc_string: None,
        }
    }
}

pub struct NodeWithMetadata<'a> {
    pub category: Category,
    pub node: NodeInfo<'a>,
}

#[derive(Debug)]
pub struct TensorInfo {
    pub dims: Vec<i64>,
    pub dtype: tensor_proto::DataType,
}
