use rmlk_schema::onnx::{
    Category, NodeProto, NodeWithMetadata, NodeWithValue, TensorProto, ValueInfoProto,
};
use rmlk_schema::onnx_import::{OnnxGraphTraverser, Result};

pub struct FindNode<'a> {
    pub target: &'a str,
    pub node: Option<NodeWithMetadata<'a>>,
}

impl<'a> OnnxGraphTraverser<'a> for FindNode<'a> {
    fn check_input(&mut self, input: ValueInfoProto<'a>) -> Result<bool> {
        match &input.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: None,
                        tensor: None,
                        value: Some(input),
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn check_output(&mut self, output: ValueInfoProto<'a>) -> Result<bool> {
        match &output.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: None,
                        tensor: None,
                        value: Some(output),
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn check_initializer(&mut self, initializer: TensorProto<'a>) -> Result<bool> {
        match &initializer.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: None,
                        tensor: Some(initializer),
                        value: None,
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn check_inner_node(&mut self, node: NodeProto<'a>) -> Result<bool> {
        match &node.name {
            Some(name) if name == self.target => {
                self.node = Some(NodeWithMetadata {
                    category: Category::Input,
                    node_with_value: NodeWithValue {
                        node: Some(node),
                        tensor: None,
                        value: None,
                    },
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
