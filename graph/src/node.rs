use rmlk_ir::{Attribute, DataType, Op, ValueInfo};
use std::collections::HashMap;

// Todo: Sometimes we dont want to keep all of a Definition specially in release
// because we only need certain things and do not need the metadata.
// Let's solve it.

pub struct Node {
    /// The node's Provider.
    ///
    /// Each node is assigned to a single Provider.
    _provider: Option<u32>,
    /// The node's operation.
    ///
    /// If the op is NoOp, this node is an input and graph leaf.
    op: Op,
    /// Inputs for this node.
    ///
    /// An ID may correspond to a node or
    /// an initial tensor.
    inputs: Vec<usize>,
    /// Outputs for this node.
    ///
    /// An ID may correspond to a node or
    /// an initial tensor.
    outputs: Vec<usize>,
    /// The node's definition.
    _definition: Definition,
    /// Attributes.
    attributes: HashMap<Box<str>, Attribute>,
}

impl Node {
    pub fn new(op: Op, definition: Definition) -> Self {
        Self {
            op,
            _provider: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            _definition: definition,
            attributes: HashMap::new(),
        }
    }

    pub fn op(&self) -> Op {
        self.op
    }

    pub fn inputs(&self) -> &[usize] {
        self.inputs.as_slice()
    }

    pub fn add_input(&mut self, node_id: usize) {
        self.inputs.push(node_id);
    }

    pub fn outputs(&self) -> &[usize] {
        self.outputs.as_slice()
    }

    pub fn add_output(&mut self, node_id: usize) {
        self.outputs.push(node_id);
    }

    pub fn attrs(&self) -> &HashMap<Box<str>, Attribute> {
        &self.attributes
    }

    pub fn add_attr(&mut self, name: Box<str>, attr: Attribute) {
        self.attributes.insert(name, attr);
    }

    pub fn def(&self) -> &Definition {
        &self._definition
    }
}

#[derive(Default)]
pub struct Definition {
    value: Option<ValueInfo>,
    node: Option<rmlk_ir::Node>,
    header: Option<TensorHeader>,
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

    pub fn node(node: rmlk_ir::Node) -> Self {
        Self {
            node: Some(node),
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
}

struct TensorHeader {
    pub name: String,
    pub dtype: DataType,
    pub dims: Vec<usize>,
}
