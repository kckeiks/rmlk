use log::warn;
use rmlk_graph::{Definition, GraphBuilder, Node};
use rmlk_ir::{DataType, Graph, Op};
use rmlk_tensor::provider::{Error as ProviderError, Provider};
use std::alloc::Allocator;
use std::collections::HashMap;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Device(ProviderError),
    MissingInputNode,
    Unknown,
}

pub fn parse_ir_graph<A: Allocator + Clone>(
    graph_schema: Graph,
    alloc: A,
) -> Result<rmlk_graph::Graph> {
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
            Vec::new(),
            Vec::new(),
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
            Vec::new(),
            Vec::new(),
            Definition {
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

    for mut ir_node in graph_schema.node {
        let op = ir_node
            .op_type
            .map(|op| op.parse::<Op>())
            .ok_or(Error::Unknown)
            .unwrap()
            .map_err(|_| Error::Unknown)
            .unwrap();
        let mut node = Node::new(
            op,
            Vec::new(),
            Vec::new(),
            Definition {
                shape: Vec::new(),
                dtype: DataType::Undefined,
            },
        );

        for name in ir_node.input.iter() {
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
            Vec::new(),
            Vec::new(),
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

#[cfg(test)]
mod test {
    use quick_protobuf::{BytesReader, MessageRead};
    use rmlk_ir::ModelProto;
    use std::fs;
    use std::io::Read;

    #[test]
    fn test_load_model() {
        let model = fs::read(
            "/Users/acadia/Repo/notebooks/resnet34/model.resnet34.with.external.data.onnx",
        )
        .expect("bad");
        let mut reader = BytesReader::from_bytes(&model);
        let model_proto = ModelProto::from_reader(&mut reader, &model).unwrap();

        println!("{:?}\n", model_proto.graph.as_ref().unwrap().initializer);
        println!(
            "{:?}\n",
            model_proto.graph.as_ref().unwrap().node.get(0).unwrap()
        );
        println!(
            "{:?}\n",
            model_proto.graph.as_ref().unwrap().node.get(1).unwrap()
        );
        println!(
            "{:?}\n",
            model_proto.graph.as_ref().unwrap().node.get(2).unwrap()
        );
        println!(
            "{:?}\n",
            model_proto.graph.as_ref().unwrap().node.get(3).unwrap()
        );
        println!(
            "{:?}\n",
            model_proto.graph.as_ref().unwrap().node.last().unwrap()
        );
        println!("{:?}", model_proto.graph.unwrap().output);
    }
}
