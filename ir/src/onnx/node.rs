use crate::error::Error;
use crate::{dimension_proto, tensor_proto, ty_proto, NodeProto, TensorProto, ValueInfoProto};
use std::fmt::{Debug, Formatter};

#[derive(Debug)]
pub enum Category {
    Input,
    Output,
    Initializer,
    InnerNode,
}

pub struct NodeWithMetadata<'a> {
    pub category: Category,
    pub node_with_value: NodeWithValue<'a>,
}

impl NodeWithMetadata<'_> {
    pub fn name(&self) -> Option<&str> {
        let name_from_node = self
            .node_with_value
            .node
            .as_ref()
            .map(|node| node.name.as_ref())
            .flatten()
            .map(|name| name.as_ref());

        if name_from_node.is_some() {
            name_from_node
        } else {
            self.node_with_value
                .value
                .as_ref()
                .map(|node| node.name.as_ref())
                .flatten()
                .map(|name| name.as_ref())
        }
    }
}

impl Debug for NodeWithMetadata<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("NodeWithMetadata");
        debug_struct.field("category", &self.category);

        if let Some(name) = &self
            .node_with_value
            .node
            .as_ref()
            .map(|n| n.name.as_ref())
            .flatten()
        {
            debug_struct.field("name", name);
        } else {
            debug_struct.field("name", &"unknown");
        }

        if self.node_with_value.tensor.is_some() {
            if let Some(tensor_info) = &self.node_with_value.tensor {
                debug_struct.field("dtype", &tensor_info.data_type);
                debug_struct.field("dimensions", &tensor_info.dims);
            }
        } else if self.node_with_value.value.is_some() {
            if let Some(value_info_proto) = &self.node_with_value.value {
                // Todo: let's compute this once because this is expensive.
                let tensor_info =
                    ValueInfo::try_from(value_info_proto).map_err(|_| std::fmt::Error)?;
                debug_struct.field("dtype", &tensor_info.dtype);
                debug_struct.field("dimensions", &tensor_info.dims);
            }
        }

        if let Some(input) = &self
            .node_with_value
            .node
            .as_ref()
            .map(|n| n.input.as_slice())
        {
            debug_struct.field("input", input);
        }

        if let Some(output) = &self
            .node_with_value
            .node
            .as_ref()
            .map(|n| n.output.as_slice())
        {
            debug_struct.field("output", output);
        }

        debug_struct.finish()
    }
}

pub struct NodeWithValue<'a> {
    node: Option<NodeProto<'a>>,
    tensor: Option<TensorProto<'a>>,
    value: Option<ValueInfoProto<'a>>,
}

impl<'a> TryFrom<NodeProto<'a>> for NodeWithValue<'a> {
    type Error = Error;

    fn try_from(value: NodeProto<'a>) -> Result<Self, Self::Error> {
        Ok(NodeWithValue {
            node: Some(value),
            tensor: None,
            value: None,
        })
    }
}

impl<'a> From<TensorProto<'a>> for NodeWithValue<'a> {
    fn from(value: TensorProto<'a>) -> Self {
        NodeWithValue {
            node: None,
            tensor: Some(value),
            value: None,
        }
    }
}

impl<'a> From<ValueInfoProto<'a>> for NodeWithValue<'a> {
    fn from(value: ValueInfoProto<'a>) -> Self {
        NodeWithValue {
            node: None,
            tensor: None,
            value: Some(value),
        }
    }
}

#[derive(Debug)]
pub struct ValueInfo {
    dims: Vec<i64>,
    dtype: tensor_proto::DataType,
}

impl TryFrom<&ValueInfoProto<'_>> for ValueInfo {
    type Error = Error;

    fn try_from(value: &ValueInfoProto<'_>) -> Result<Self, Self::Error> {
        let type_proto = value.type_pb.as_ref().ok_or(Error::InvalidValue {
            field: "".to_string(),
            value: "".to_string(),
        })?;
        let info = match &type_proto.value {
            ty_proto::OneOfvalue::tensor_type(ty_proto::Tensor { elem_type, shape }) => {
                let dtype = tensor_proto::DataType::from(elem_type.unwrap_or(0));
                // Todo: Let's remove this.
                let mut dims = Vec::new();
                if let Some(tensor_shape_proto) = shape {
                    for dimension in &tensor_shape_proto.dim {
                        match dimension.value {
                            dimension_proto::OneOfvalue::dim_value(val) => {
                                dims.push(val);
                            }
                            dimension_proto::OneOfvalue::dim_param(_) => {}
                            dimension_proto::OneOfvalue::None => {}
                        }
                    }
                }
                ValueInfo { dims, dtype }
            }
            ty_proto::OneOfvalue::sequence_type(_) => unimplemented!(),
            ty_proto::OneOfvalue::map_type(_) => unimplemented!(),
            ty_proto::OneOfvalue::optional_type(_) => unimplemented!(),
            ty_proto::OneOfvalue::sparse_tensor_type(_) => unimplemented!(),
            ty_proto::OneOfvalue::None => unimplemented!(),
        };

        Ok(info)
    }
}
