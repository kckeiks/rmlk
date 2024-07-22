use log::warn;
use rmlk_graph::{Definition, GraphBuilder, Node};
use rmlk_ir::{DataType, Graph, Op};
use rmlk_tensor::Error as ProviderError;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Device(ProviderError),
    MissingInputNode,
    Unknown,
}

pub fn parse_ir_graph(graph_schema: Graph) -> Result<rmlk_graph::Graph> {
    let mut builder = GraphBuilder::new();

    for value_info in graph_schema.input {
        let ty = value_info.ty.ok_or(Error::Unknown).unwrap();
        // Todo: Should we support other types such as maps, sparse tensors, etc.
        let (elem_ty, shape) = ty.get_tensor_info().ok_or(Error::Unknown).unwrap();

        // Todo: Add allocator API to schema.
        let mut shape_ = Vec::new();
        if let Some(s) = shape {
            shape_.extend(s);
        }

        let node = Node::new(
            Op::NoOp,
            Definition {
                shape: shape_,
                dtype: elem_ty,
            },
        );
        let node_id = builder.add_input(node).expect("TODO");

        if let Some(old_id) = builder.insert_name_to_id(value_info.name, node_id) {
            // Todo: Rename name.
            warn!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }
    }

    for mut tensor in graph_schema.initializer {
        let elem_ty = tensor.data_type.try_into().unwrap();
        let name = tensor.name.take().ok_or(Error::MissingInputNode).unwrap();

        let mut node = Node::new(
            Op::Const,
            Definition {
                // Todo: we need to read the dimensions.
                shape: Vec::new(),
                dtype: elem_ty,
            },
        );

        let initial_id = builder.add_initial_tensor(tensor);
        // It's a Const node so it's input is an id to an initial tensor.
        node.add_input(initial_id);

        let node_id = builder.add_node(node).expect("TODO");
        if let Some(old_id) = builder.insert_name_to_id(name, node_id) {
            // Todo: Rename name.
            warn!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }
    }

    for ir_node in graph_schema.node {
        let op = ir_node
            .op_type
            .as_ref()
            .map(|op| op.parse::<Op>())
            .ok_or(Error::Unknown)
            .unwrap()
            .map_err(|_| Error::Unknown)
            .unwrap();
        // if matches!(op, Op::Conv) {
        //     println!("{ir_node:?}");
        // }

        let mut node = Node::new(
            op,
            Definition {
                shape: Vec::new(),
                dtype: DataType::Undefined,
            },
        );

        // Add attributes.
        for attr in ir_node.attribute {
            // Todo: Let's avoid the copy.
            // Maybe we can define some type of object that we agree to never drop
            // and then we can leak the string.
            node.add_attr(attr.name.clone().into_boxed_str(), attr);
        }

        for name in ir_node.input.iter() {
            // Todo: Mapping one name to a single node id, we lose information,
            // because a single node might have two outputs, how do we differentiate?
            let input_node_id = builder.get_node_id(name).ok_or_else(|| {
                println!("name {name}");
                Error::MissingInputNode
            })?;
            node.add_input(input_node_id);
        }

        let node_id = builder.add_node(node).expect("TODO");

        for name in ir_node.output {
            builder.insert_name_to_id(name, node_id);
        }
    }

    for value_info in graph_schema.output {
        let ty = value_info.ty.ok_or(Error::Unknown)?;
        // Todo: Should we support other types such as maps, sparse tensors, etc.
        let (elem_ty, shape) = ty.get_tensor_info().ok_or(Error::Unknown)?;

        // Todo: Add allocator API to schema.
        let mut shape_ = Vec::new();
        if let Some(s) = shape {
            shape_.extend(s);
        }

        let node = Node::new(
            Op::NoOp,
            Definition {
                shape: shape_,
                dtype: elem_ty,
            },
        );
        let node_id = builder.add_output(node).expect("TODO");

        if let Some(old_id) = builder.insert_name_to_id(value_info.name, node_id) {
            // Todo: Rename name.
            warn!("found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }
    }

    builder.build().map_err(|_| Error::Unknown)
}
