use crate::Op;
use crate::{Attribute, TypeValue};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Deserialize, Serialize)]
pub struct Node {
    // Input nodes.
    pub input: Option<Vec<usize>>,
    // Output nodes.
    pub output: Option<Vec<usize>>,
    // An identifier for this node in a graph.
    pub id: usize,
    // The symbolic identifier of the Operator to execute.
    pub op_type: Op,
    // Additional named attributes.
    pub attribute: Option<Vec<Attribute>>,
    pub value: Option<TypeValue>,
    #[cfg(debug_assertions)]
    // Optional name of node.
    pub name: Option<String>,
}

impl Node {
    pub fn new(id: usize) -> Self {
        Self {
            input: None,
            output: None,
            id,
            op_type: Op::NoOp,
            attribute: None,
            value: None,
            #[cfg(debug_assertions)]
            name: None,
        }
    }

    pub fn add_input(&mut self, input: usize) {
        if self.input.is_none() {
            let _ = self.input.insert(vec![input]);
        } else {
            self.input
                .as_mut()
                .expect("That we initialize first before modifying")
                .push(input);
        }
    }

    pub fn add_output(&mut self, output: usize) {
        if self.output.is_none() {
            let _ = self.output.insert(vec![output]);
        } else {
            self.output
                .as_mut()
                .expect("That we initialize first before modifying")
                .push(output);
        }
    }

    pub fn set_inputs(&mut self, inputs: Vec<usize>) {
        self.input = Some(inputs);
    }

    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.output = Some(outputs);
    }

    pub fn op(&self) -> Op {
        self.op_type
    }

    pub fn set_op(&mut self, op: Op) {
        self.op_type = op;
    }

    pub fn set_type_value(&mut self, value: TypeValue) {
        self.value = Some(value);
    }

    pub fn set_attributes(&mut self, attrs: Vec<Attribute>) {
        self.attribute = Some(attrs);
    }

    #[cfg(debug_assertions)]
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }
}
