use rmlk_graph::{Definition, GraphBuilder, Node, OnnxGraphTraverser, TraversalError};
use rmlk_ir::{Category, NodeWithMetadata, Op, ValueInfo};

pub struct ExecutionGraphFromOnnx {
    builder: GraphBuilder,
}

impl<'a> OnnxGraphTraverser<'a> for ExecutionGraphFromOnnx {
    fn check_node(&mut self, node: NodeWithMetadata<'a>) -> Result<bool, TraversalError> {
        match node.category {
            Category::Input => {
                let node_name = node.name().map(str::to_string);
                let value_proto = node
                    .node_with_value
                    .value
                    .as_ref()
                    .ok_or(TraversalError::MissingValue)?;
                let value = ValueInfo::try_from(value_proto)
                    .map_err(|e| TraversalError::TransformationFailed)?;
                let final_node = Node::new(
                    Op::NoOp,
                    Definition {
                        // Todo: Fix.
                        shape: value.dims.iter().copied().map(|n| n as usize).collect(),
                        dtype: value.dtype.into(),
                        node: None,
                        // Todo: Fix.
                        name: node.name().map(str::to_string).unwrap_or(String::new()),
                    },
                );
                let node_id = self.builder.add_input(final_node).unwrap();

                if let Some(old_id) = self
                    .builder
                    .insert_name_to_id(node_name.unwrap_or(String::new()), node_id)
                {
                    // Todo: Rename name.
                    println!(
                        "found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]"
                    );
                }
            }
            Category::Output => {}
            Category::Initializer => {}
            Category::InnerNode => {}
        }

        Ok(false)
    }
}
