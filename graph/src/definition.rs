use rmlk_ir::{Attribute, DataType, ValueInfo};
use std::collections::HashMap;

#[derive(Default)]
pub struct Definition {
    value: Option<ValueInfo>,
    node: Option<rmlk_ir::Node>,
    header: Option<TensorHeader>,
    attributes: Option<HashMap<Box<str>, Attribute>>,
}

impl Definition {
    pub fn tensor(name: String, dtype: DataType, dims: Vec<usize>) -> Self {
        Self {
            header: Some(TensorHeader {
                // Todo: remove clone.
                name,
                dtype,
                dims,
            }),
            ..Default::default()
        }
    }

    pub fn value(value: ValueInfo) -> Self {
        Self {
            value: Some(value),
            ..Default::default()
        }
    }

    pub fn node(mut node: rmlk_ir::Node) -> Self {
        let mut attributes = None;

        if !node.attribute.is_empty() {
            // Add attributes.
            let mut attrs = HashMap::new();
            for attr in std::mem::take(&mut node.attribute) {
                // Todo: Let's avoid the copy.
                // Maybe we can define some type of object that we agree to never drop
                // and then we can leak the string.
                attrs.insert(attr.name.clone().into_boxed_str(), attr);
            }
            attributes = Some(attrs);
        }

        Self {
            node: Some(node),
            attributes,
            ..Default::default()
        }
    }

    pub fn set_value(&mut self, value: ValueInfo) -> Option<ValueInfo> {
        self.value.replace(value)
    }

    pub fn set_node(&mut self, node: rmlk_ir::Node) -> Option<rmlk_ir::Node> {
        self.node.replace(node)
    }

    // Todo: remove clones in this method.
    pub fn shape(&self) -> Option<Vec<usize>> {
        if let Some(value) = self.value.as_ref() {
            // Todo: `get_tensor_info` should not return None at this point.
            // Consider dynamic size tensors.
            let (_, dims) = value.ty.as_ref()?.get_tensor_info()?;
            return dims;
        }

        if let Some(tensor) = self.header.as_ref() {
            return Some(tensor.dims.clone());
        }

        // Todo: circle back.
        // This is an inner node and we don't get that information from schema.
        if self.node.is_some() {
            return None;
        }

        None
    }

    pub fn dtype(&self) -> Option<DataType> {
        if let Some(value) = self.value.as_ref() {
            // Todo: `get_tensor_info` should not return None at this point.
            // Consider dynamic size tensors.
            let (dtype, _) = value.ty.as_ref()?.get_tensor_info()?;
            return Some(dtype);
        }

        if let Some(tensor) = self.header.as_ref() {
            return Some(tensor.dtype);
        }

        // Todo: circle back.
        // This is an inner node and we don't get that information from schema.
        if self.node.is_some() {
            return Some(DataType::Undefined);
        }

        Some(DataType::Undefined)
    }

    pub fn name(&self) -> Option<&str> {
        if let Some(value) = self.value.as_ref() {
            return Some(value.name.as_str());
        }

        if let Some(tensor) = self.header.as_ref() {
            return Some(tensor.name.as_str());
        }

        // Todo: circle back.
        // This is an inner node and we don't get that information from schema.
        if self.node.is_some() {
            unreachable!("value should exist for nodes");
        }

        None
    }

    pub fn inputs(&self) -> Option<&Vec<String>> {
        self.node.as_ref().map(|n| &n.input)
    }

    pub fn outputs(&self) -> Option<&Vec<String>> {
        self.node.as_ref().map(|n| &n.output)
    }

    pub fn attrs(&self) -> Option<&HashMap<Box<str>, Attribute>> {
        self.attributes.as_ref()
    }
}

struct TensorHeader {
    pub name: String,
    pub dtype: DataType,
    pub dims: Vec<usize>,
}
