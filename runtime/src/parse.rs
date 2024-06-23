use log::warn;
use rmlk_graph::device::{CpuDevice, Device, DeviceError};
use rmlk_graph::{GraphBuilder, Node, Op};
use rmlk_hir::{DataType, Graph};
use std::collections::HashMap;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Device(DeviceError),
    MissingInputNode,
    Unknown,
}

pub fn parse_ir_graph(ir_graph: Graph) -> Result<rmlk_graph::Graph<CpuDevice>> {
    let device = CpuDevice;
    let mut builder = GraphBuilder::new(device.clone());

    for value_info in ir_graph.input {
        let ty = value_info.ty.ok_or(Error::Unknown).unwrap();

        let (elem_ty, shape) = ty.get_tensor_info().ok_or(Error::Unknown).unwrap();
        let tensor = if let Some(shape) = shape {
            device
                .tensor_from_dtype_with_shape(elem_ty, shape)
                .map_err(Error::Device)?
        } else {
            device.tensor_from_dtype(elem_ty).map_err(Error::Device)?
        };

        let node = Node::new_with_tensor(Op::NoOp, device.clone(), tensor);
        let node_id = builder.add_input(node).expect("TODO");

        if let Some(old_id) = builder.store_id_by_name(value_info.name, node_id) {
            // Todo: Rename name.
            warn!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }
    }

    for tensor in ir_graph.initializer {
        let elem_ty = tensor.data_type.try_into().unwrap();
        let name = tensor.name.ok_or(Error::MissingInputNode).unwrap();
        let tensor = device
            .tensor_from_dtype_with_shape(elem_ty, tensor.dims)
            .map_err(Error::Device)?;

        let node = Node::new_with_tensor(Op::NoOp, device.clone(), tensor);
        let node_id = builder.add_input(node).expect("TODO");

        if let Some(old_id) = builder.store_id_by_name(name, node_id) {
            // Todo: Rename name.
            warn!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }
    }

    for mut ir_node in ir_graph.node {
        let op = ir_node
            .op_type
            .map(|op| op.parse::<Op>())
            .ok_or(Error::Unknown)
            .unwrap()
            .map_err(|_| Error::Unknown)
            .unwrap();
        let mut node = Node::new(op, device.clone());

        for name in ir_node.input.iter() {
            let input_node_id = builder.get_store_id_by_name(name).ok_or_else(|| {
                println!("name {name}");
                Error::MissingInputNode
            })?;
            node.set_input(input_node_id);
        }

        let node_id = builder.add_node(node).expect("TODO");

        for name in ir_node.output {
            builder.store_id_by_name(name, node_id);
        }
    }

    for value_info in ir_graph.output {
        let ty = value_info.ty.ok_or(Error::Unknown)?;

        // Todo: Should we support other types such as maps, sparse tensors, etc.
        let (elem_ty, shape) = ty.get_tensor_info().ok_or(Error::Unknown)?;
        let tensor = if let Some(shape) = shape {
            device
                .tensor_from_dtype_with_shape(elem_ty, shape)
                .map_err(Error::Device)?
        } else {
            device.tensor_from_dtype(elem_ty).map_err(Error::Device)?
        };

        let node = Node::new_with_tensor(Op::NoOp, device.clone(), tensor);
        let node_id = builder.add_output(node).expect("TODO");

        if let Some(old_id) = builder.store_id_by_name(value_info.name, node_id) {
            // Todo: Rename name.
            warn!("found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]");
        }
    }

    builder.build().map_err(|_| Error::Unknown)
}

#[cfg(test)]
mod test {
    use quick_protobuf::{BytesReader, MessageRead};
    use rmlk_hir::ModelProto;
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
