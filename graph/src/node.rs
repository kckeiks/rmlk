use crate::definition::Definition;

// Todo: Sometimes we dont want to keep all of a Definition specially in release
// because we only need certain things and do not need the metadata.
// Let's solve it.
pub struct Node {
    /// The node's Provider.
    ///
    /// Each node is assigned to a single Provider.
    _provider: Option<u32>,
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
    definition: Definition,
}

impl Node {
    pub fn new(definition: Definition) -> Self {
        Self {
            _provider: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            definition,
        }
    }

    pub fn from_definition(mut definition: Definition) -> Self {
        Self {
            _provider: None,
            inputs: definition.take_inputs().unwrap_or_default(),
            outputs: definition.take_outputs().unwrap_or_default(),
            definition,
        }
    }

    pub fn inputs(&self) -> &[usize] {
        self.inputs.as_slice()
    }

    pub fn add_input(&mut self, node_id: usize) {
        self.inputs.push(node_id);
    }

    pub fn set_input(&mut self, inputs: Vec<usize>) {
        self.inputs = inputs;
    }

    pub fn outputs(&self) -> &[usize] {
        self.outputs.as_slice()
    }

    pub fn add_output(&mut self, node_id: usize) {
        self.outputs.push(node_id);
    }

    pub fn set_output(&mut self, outputs: Vec<usize>) {
        self.outputs = outputs;
    }

    pub fn def(&self) -> &Definition {
        &self.definition
    }
}
