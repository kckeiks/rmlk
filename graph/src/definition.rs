use rmlk_ir::ir_v2::{Attribute, Node};
use rmlk_ir::{DataType, ValueInfo};
use std::collections::HashMap;

pub struct Definition {
    // value: Option<ValueInfo>,
    // node: Option<rmlk_ir::Node>,
    // header: Option<TensorHeader>,
    // attributes: Option<HashMap<Box<str>, Attribute>>,
    node: Node,
    attributes: Option<HashMap<Box<str>, Attribute>>,
}

impl Definition {
    pub fn new(node: Node) -> Self {
        let mut node = node;
        let mut attributes = None;

        if let Some(attrs) = &mut node.attribute {
            if !attrs.is_empty() {
                // Add attributes.
                let mut attrs_map = HashMap::new();
                for attr in std::mem::take(attrs) {
                    // Todo: Let's avoid the copy.
                    // Maybe we can define some type of object that we agree to never drop
                    // and then we can leak the string.
                    attrs_map.insert(attr.name.clone().into_boxed_str(), attr);
                }
                attributes = Some(attrs_map);
            }
        }

        Self { node, attributes }
    }
    pub fn tensor(name: String, dtype: DataType, dims: Vec<usize>) -> Self {
        unimplemented!()
        // Self {
        //     header: Some(TensorHeader {
        //         // Todo: remove clone.
        //         name,
        //         dtype,
        //         dims,
        //     }),
        //     ..Default::default()
        // }
    }

    pub fn value(value: ValueInfo) -> Self {
        // Self {
        //     value: Some(value),
        //     ..Default::default()
        // }
        unimplemented!()
    }

    pub fn node(mut node: rmlk_ir::Node) -> Self {
        // let mut attributes = None;
        //
        // if !node.attribute.is_empty() {
        //     // Add attributes.
        //     let mut attrs = HashMap::new();
        //     for attr in std::mem::take(&mut node.attribute) {
        //         // Todo: Let's avoid the copy.
        //         // Maybe we can define some type of object that we agree to never drop
        //         // and then we can leak the string.
        //         attrs.insert(attr.name.clone().into_boxed_str(), attr);
        //     }
        //     attributes = Some(attrs);
        // }
        //
        // Self {
        //     node: Some(node),
        //     attributes,
        //     ..Default::default()
        // }
        unimplemented!()
    }

    // pub fn set_value(&mut self, value: ValueInfo) -> Option<ValueInfo> {
    //     // self.value.replace(value)
    // }

    // pub fn set_node(&mut self, node: rmlk_ir::Node) -> Option<rmlk_ir::Node> {
    //     self.node.replace(node)
    // }

    // Todo: remove clones in this method.
    pub fn shape(&self) -> Option<Vec<usize>> {
        // if let Some(value) = self.value.as_ref() {
        //     // Todo: `get_tensor_info` should not return None at this point.
        //     // Consider dynamic size tensors.
        //     let (_, dims) = value.ty.as_ref()?.get_tensor_info()?;
        //     return dims;
        // }
        //
        // if let Some(tensor) = self.header.as_ref() {
        //     return Some(tensor.dims.clone());
        // }
        //
        // // Todo: circle back.
        // // This is an inner node and we don't get that information from schema.
        // if self.node.is_some() {
        //     return None;
        // }
        //
        // None
        self.node.value.as_ref().map(|v| v.dims().clone())
    }

    pub fn dtype(&self) -> Option<DataType> {
        // if let Some(value) = self.value.as_ref() {
        //     // Todo: `get_tensor_info` should not return None at this point.
        //     // Consider dynamic size tensors.
        //     let (dtype, _) = value.ty.as_ref()?.get_tensor_info()?;
        //     return Some(dtype);
        // }
        //
        // if let Some(tensor) = self.header.as_ref() {
        //     return Some(tensor.dtype);
        // }
        //
        // // Todo: circle back.
        // // This is an inner node and we don't get that information from schema.
        // if self.node.is_some() {
        //     return Some(DataType::Undefined);
        // }
        //
        // Some(DataType::Undefined)
        // Todo: Handle unwrap().
        self.node.value.as_ref().map(|v| DataType::try_from(v.ty()).unwrap())
    }

    pub fn name(&self) -> Option<&str> {
        // if let Some(value) = self.value.as_ref() {
        //     return Some(value.name.as_str());
        // }
        //
        // if let Some(tensor) = self.header.as_ref() {
        //     return Some(tensor.name.as_str());
        // }
        //
        // // Todo: circle back.
        // // This is an inner node and we don't get that information from schema.
        // if self.node.is_some() {
        //     unreachable!("value should exist for nodes");
        // }
        //
        // None
        self.node.name.as_ref().map(|name| name.as_str())
    }

    pub fn inputs(&self) -> Option<&Vec<usize>> {
        // self.node.as_ref().map(|n| &n.input)
        self.node.input.as_ref()
    }

    pub fn take_inputs(&mut self) -> Option<Vec<usize>> {
        self.node.input.take()
    }

    pub fn outputs(&self) -> Option<&Vec<usize>> {
        self.node.output.as_ref()
    }

    pub fn take_outputs(&mut self) -> Option<Vec<usize>> {
        self.node.output.take()
    }

    pub fn attrs(&self) -> Option<&HashMap<Box<str>, Attribute>> {
        self.attributes.as_ref()
    }

    pub fn op(&self) -> u32 {
        self.node.op_type
    }
}

struct TensorHeader {
    pub name: String,
    pub dtype: DataType,
    pub dims: Vec<usize>,
}
