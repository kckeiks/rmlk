use rmlk_graph::Definition;
use rmlk_graph::{Graph, GraphBuilder, GraphTraverser, Node, TraversalError};
use rmlk_ir::{Op, Tensor, ValueInfo};

pub struct ExecutionGraphBuilder {
    builder: GraphBuilder,
}

impl ExecutionGraphBuilder {
    pub fn new() -> Self {
        Self {
            builder: GraphBuilder::new(),
        }
    }

    pub fn build(self) -> Graph {
        self.builder.build().unwrap()
    }
}

impl GraphTraverser for ExecutionGraphBuilder {
    fn check_input(&mut self, input: ValueInfo) -> Result<bool, TraversalError> {
        let name = input.name.clone();

        let def = Definition::value(input);

        let final_node = Node::new(Op::NoOp, def);
        let node_id = self.builder.add_input(final_node).unwrap();

        if let Some(old_id) = self.builder.insert_name_to_id(name, node_id) {
            // Todo: Rename name.
            println!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }

        Ok(false)
    }

    fn check_output(&mut self, output: ValueInfo) -> Result<bool, TraversalError> {
        let name = output.name.clone();

        let def = Definition::value(output);

        let node = Node::new(Op::NoOp, def);
        let node_id = self.builder.add_output_node(node).expect("TODO");

        if let Some(old_id) = self.builder.insert_name_to_id(name, node_id) {
            // Todo: Rename name.
            println!("found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }

        Ok(false)
    }

    fn check_initializer(&mut self, initializer: Tensor) -> Result<bool, TraversalError> {
        let name = initializer
            .name
            .clone()
            .ok_or(TraversalError::InvalidTensor)?;

        let def = Definition::tensor(
            name.clone(),
            initializer.data_type,
            // Todo: remove clone.
            initializer.dims.clone(),
        );
        let node = Node::new(Op::Const, def);

        let node_id = self.builder.add_node(node).expect("TODO");
        self.builder.add_initial_tensor(node_id, initializer);

        if let Some(old_id) = self.builder.insert_name_to_id(name, node_id) {
            // Todo: Rename name.
            println!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }

        Ok(false)
    }

    fn check_inner_node(&mut self, node: rmlk_ir::Node) -> Result<bool, TraversalError> {
        unimplemented!()
        // let op = node
        //     .op_type
        //     .as_ref()
        //     .map(|op| op.parse::<Op>())
        //     .ok_or(TraversalError::InvalidInnerNode)?
        //     .map_err(|_| TraversalError::InvalidInnerNode)?;
        //
        // let def = Definition::node(node);
        // let mut node = Node::new(op, def);
        //
        // let mut inputs = Vec::new();
        // for name in node.def().inputs().ok_or(TraversalError::MissingValue)? {
        //     // Todo: Mapping one name to a single node id, we lose information,
        //     // because a single node might have two outputs, how do we differentiate?
        //     let input_node_id = self
        //         .builder
        //         .get_node_id(name)
        //         .ok_or_else(|| TraversalError::InvalidInnerNode)?;
        //     inputs.push(input_node_id);
        // }
        //
        // node.set_input(inputs);
        //
        // let node_id = self.builder.add_node(node).expect("TODO");
        //
        // let mut output_node_ids = Vec::new();
        // // Todo: remove clone.
        // let outputs = self
        //     .builder
        //     .get_node(node_id)
        //     .expect("that node was just inserted")
        //     .def()
        //     .outputs()
        //     .ok_or(TraversalError::MissingValue)?
        //     .clone();
        // for name in outputs {
        //     match self.builder.get_node_id(&name) {
        //         None => {
        //             let def = Definition::value(ValueInfo {
        //                 // Todo: remove clone.
        //                 name: name.clone(),
        //                 ty: None,
        //                 doc_string: None,
        //                 // Todo: remove this allocation.
        //                 metadata_props: vec![],
        //             });
        //             let mut output_node = Node::new(Op::NoOp, def);
        //             output_node.add_input(node_id);
        //
        //             let output_node_id = self
        //                 .builder
        //                 .add_node(output_node)
        //                 .map_err(|_| TraversalError::InvalidInnerNode)?;
        //             output_node_ids.push(output_node_id);
        //
        //             // Todo: remove clone.
        //             self.builder.insert_name_to_id(name.clone(), output_node_id);
        //         }
        //         Some(id) => {
        //             let node = self
        //                 .builder
        //                 .get_node_mut(id)
        //                 .ok_or(TraversalError::InvalidInnerNode)?;
        //             node.add_input(node_id);
        //             output_node_ids.push(id);
        //         }
        //     }
        // }
        //
        // let node = self
        //     .builder
        //     .get_node_mut(node_id)
        //     .expect("We just inserted it above.");
        // for id in output_node_ids {
        //     node.add_output(id);
        // }
        //
        // Ok(false)
    }
}
