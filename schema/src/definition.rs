use crate::{Attribute, DataType, Node, Op};
use std::collections::HashMap;

pub struct Definition {
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

    // Todo: remove clones in this method.
    pub fn shape(&self) -> Option<&Vec<usize>> {
        self.node.value.as_ref().map(|v| v.dims())
    }

    pub fn dtype(&self) -> Option<DataType> {
        self.node
            .value
            .as_ref()
            .map(|v| DataType::try_from(v.ty()).unwrap())
    }

    pub fn name(&self) -> Option<&str> {
        self.node.name.as_ref().map(|name| name.as_str())
    }

    pub fn inputs(&self) -> Option<&Vec<usize>> {
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

    pub fn op(&self) -> Op {
        self.node.op()
    }
}
