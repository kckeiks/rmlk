use rmlk_graph::{
    Definition, GraphBuilder, GraphTraverser, Node, OnnxGraphTraverser, TraversalError,
};
use rmlk_ir::{Category, DataType, Graph, NodeWithMetadata, Op, Tensor, ValueInfo, ValueInfoV2};

pub struct GraphFromOnnx {
    inner: Graph,
}

impl GraphFromOnnx {
    pub fn new() -> Self {
        Self {
            inner: Graph::default(),
        }
    }
}

impl<'a> OnnxGraphTraverser<'a> for GraphFromOnnx {
    fn check_node(&mut self, node: NodeWithMetadata<'a>) -> Result<bool, TraversalError> {
        match node.category {
            Category::Input => {
                let value_info = ValueInfo::try_from(
                    node.node_with_value
                        .value
                        .ok_or(TraversalError::MissingValue)?,
                )
                .map_err(|_| TraversalError::TransformationFailed)?;
                self.inner.input.push(value_info);
            }
            Category::Output => {
                let value_info = ValueInfo::try_from(
                    node.node_with_value
                        .value
                        .ok_or(TraversalError::MissingValue)?,
                )
                .map_err(|_| TraversalError::TransformationFailed)?;
                self.inner.output.push(value_info);
            }
            Category::Initializer => {
                let tensor = Tensor::from_onnx_tensor(
                    node.node_with_value
                        .tensor
                        .ok_or(TraversalError::MissingValue)?,
                )
                .map_err(|_| TraversalError::TransformationFailed)?;

                self.inner.initializer.push(tensor);
            }
            Category::InnerNode => {
                let node = rmlk_ir::Node::try_from(
                    node.node_with_value
                        .node
                        .ok_or(TraversalError::MissingValue)?,
                )
                .map_err(|_| TraversalError::TransformationFailed)?;

                self.inner.node.push(node)
            }
        }

        Ok(false)
    }
}

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
                let value = ValueInfoV2::try_from(value_proto)
                    .map_err(|_| TraversalError::TransformationFailed)?;
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

pub struct ExecutionGraphBuilder {
    builder: GraphBuilder,
}

impl GraphTraverser for ExecutionGraphBuilder {
    fn check_input(&mut self, input: ValueInfo) -> Result<bool, TraversalError> {
        let (dtype, dims) = input
            .ty
            .ok_or(TraversalError::MissingValue)?
            .get_tensor_info()
            .ok_or(TraversalError::MissingValue)?;

        let final_node = Node::new(
            Op::NoOp,
            Definition {
                shape: dims.ok_or(TraversalError::MissingValue)?,
                dtype,
                node: None,
                name: input.name.clone(),
            },
        );
        let node_id = self.builder.add_input(final_node).unwrap();

        if let Some(old_id) = self.builder.insert_name_to_id(input.name, node_id) {
            // Todo: Rename name.
            println!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }

        Ok(false)
    }

    fn check_output(&mut self, output: ValueInfo) -> Result<bool, TraversalError> {
        let ty = output.ty.ok_or(TraversalError::MissingType)?;
        // Todo: Should we support other types such as maps, sparse tensors, etc.
        let (elem_ty, shape) = ty.get_tensor_info().ok_or(TraversalError::Unknown)?;

        let mut shape_ = Vec::new();
        if let Some(s) = shape {
            shape_.extend(s);
        }

        let node = Node::new(
            Op::NoOp,
            Definition {
                shape: shape_,
                dtype: elem_ty,
                node: None,
                name: output.name.clone(),
            },
        );
        let node_id = self.builder.add_output_node(node).expect("TODO");

        if let Some(old_id) = self.builder.insert_name_to_id(output.name, node_id) {
            // Todo: Rename name.
            println!("found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }

        Ok(false)
    }

    fn check_initializer(&mut self, initializer: Tensor) -> Result<bool, TraversalError> {
        let elem_ty = initializer
            .data_type
            .try_into()
            .map_err(|_| TraversalError::InvalidInnerNode)?;
        let name = initializer
            .name
            .clone()
            .ok_or(TraversalError::InvalidTensor)?;

        let node = Node::new(
            Op::Const,
            Definition {
                // Todo: we need to read the dimensions.
                shape: initializer.dims.clone(),
                dtype: elem_ty,
                node: None,
                name: name.clone(),
            },
        );

        let node_id = self.builder.add_node(node).expect("TODO");
        self.builder.add_initial_tensor(node_id, initializer);

        if let Some(old_id) = self.builder.insert_name_to_id(name, node_id) {
            // Todo: Rename name.
            println!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }

        Ok(false)
    }

    fn check_inner_node(&mut self, node: rmlk_ir::Node) -> Result<bool, TraversalError> {
        let op = node
            .op_type
            .as_ref()
            .map(|op| op.parse::<Op>())
            .ok_or(TraversalError::InvalidInnerNode)?
            .map_err(|_| TraversalError::InvalidInnerNode)?;

        let mut res_node = Node::new(
            op,
            Definition {
                shape: Vec::new(),
                dtype: DataType::Undefined,
                node: Some(node.clone()),
                name: node.name.clone().unwrap_or("".to_string()),
            },
        );

        // Add attributes.
        for attr in node.attribute {
            // Todo: Let's avoid the copy.
            // Maybe we can define some type of object that we agree to never drop
            // and then we can leak the string.
            res_node.add_attr(attr.name.clone().into_boxed_str(), attr);
        }

        for name in node.input.iter() {
            // Todo: Mapping one name to a single node id, we lose information,
            // because a single node might have two outputs, how do we differentiate?
            let input_node_id = self
                .builder
                .get_node_id(name)
                .ok_or_else(|| TraversalError::InvalidInnerNode)?;
            res_node.add_input(input_node_id);
        }

        let node_id = self.builder.add_node(res_node).expect("TODO");

        let mut output_node_ids = Vec::new();
        for name in node.output {
            match self.builder.get_node_id(&name) {
                None => {
                    let mut output_node = Node::new(
                        Op::NoOp,
                        Definition {
                            name: name.clone(),
                            ..Default::default()
                        },
                    );
                    output_node.add_input(node_id);

                    let output_node_id = self
                        .builder
                        .add_node(output_node)
                        .map_err(|_| TraversalError::InvalidInnerNode)?;

                    output_node_ids.push(output_node_id);
                    self.builder.insert_name_to_id(name, output_node_id);
                }
                Some(id) => {
                    let node = self
                        .builder
                        .get_node_mut(id)
                        .ok_or(TraversalError::InvalidInnerNode)?;
                    node.add_input(node_id);
                    output_node_ids.push(id);
                }
            }
        }

        let node = self
            .builder
            .get_node_mut(node_id)
            .expect("We just inserted it above.");
        for id in output_node_ids {
            node.add_output(id);
        }

        Ok(false)
    }
}
