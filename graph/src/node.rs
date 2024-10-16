use rmlk_ir::{Attribute, DataType, Op, ValueInfo};
use std::collections::HashMap;

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
    _definition: NodeDefinition,
    // _definition: Definition,
    /// Attributes.
    attributes: HashMap<Box<str>, Attribute>,
}

impl Node {
    // pub fn new(op: Op, definition: Definition) -> Self {
    //     Self {
    //         op,
    //         _provider: None,
    //         inputs: Vec::new(),
    //         outputs: Vec::new(),
    //         _definition: definition,
    //         attributes: HashMap::new(),
    //     }
    // }

    pub fn new(op: Op, definition: NodeDefinition) -> Self {
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

    pub fn def(&self) -> &NodeDefinition {
        &self._definition
    }
}

/// The definition for this node's inputs and outputs.
pub struct Definition {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub node: Option<rmlk_ir::Node>,
    pub name: String,
}

impl Default for Definition {
    fn default() -> Self {
        Self {
            shape: Vec::new(),
            dtype: DataType::Undefined,
            node: None,
            name: "".to_string(),
        }
    }
}

#[derive(Default)]
pub struct NodeDefinition {
    value: Option<ValueInfo>,
    node: Option<rmlk_ir::Node>,
    tensor: Option<TensorHeader>,
}

impl NodeDefinition {
    pub fn set_value(&mut self, value: ValueInfo) -> Option<ValueInfo> {
        self.value.replace(value)
    }

    pub fn set_node(&mut self, node: rmlk_ir::Node) -> Option<rmlk_ir::Node> {
        self.node.replace(node)
    }

    pub fn set_tensor(&mut self, tensor: TensorHeader) -> Option<TensorHeader> {
        self.tensor.replace(tensor)
    }

    // Todo: remove clones in this method.
    pub fn shape(&self) -> Option<Vec<usize>> {
        if let Some(value) = self.value.as_ref() {
            // Todo: `get_tensor_info` should not return None at this point.
            // Consider dynamic size tensors.
            let (_, dims) = value.ty.as_ref()?.get_tensor_info()?;
            return dims;
        }

        if let Some(tensor) = self.tensor.as_ref() {
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

        if let Some(tensor) = self.tensor.as_ref() {
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

        if let Some(tensor) = self.tensor.as_ref() {
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

pub struct TensorHeader {
    pub name: String,
    pub dtype: DataType,
    pub dims: Vec<usize>,
}
