use crate::traverse::ExecutionGraphBuilder;
use rmlk_graph::{visit_graph, Definition, GraphBuilder, Node};
use rmlk_ir::{DataType, Graph, Op};

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Device,
    MissingInputNode,
    Unknown,
}

pub fn parse_ir_graph_v2(graph_schema: Graph) -> Result<rmlk_graph::Graph> {
    let mut builder = ExecutionGraphBuilder::new();
    visit_graph(graph_schema, &mut builder).unwrap();
    let graph = builder.build();
    Ok(graph)
}

pub fn parse_ir_graph(graph_schema: Graph) -> Result<rmlk_graph::Graph> {
    //     let mut builder = GraphBuilder::new();
    //
    //     for value_info in graph_schema.input {
    //         let ty = value_info.ty.ok_or(Error::Unknown).unwrap();
    //         // Todo: Should we support other types such as maps, sparse tensors, etc.
    //         let (elem_ty, shape) = ty.get_tensor_info().ok_or(Error::Unknown).unwrap();
    //
    //         // Todo: Add allocator API to schema.
    //         let mut shape_ = Vec::new();
    //         if let Some(s) = shape {
    //             shape_.extend(s);
    //         }
    //
    //         let node = Node::new(
    //             Op::NoOp,
    //             Definition {
    //                 shape: shape_,
    //                 dtype: elem_ty,
    //                 node: None,
    //                 name: value_info.name.clone(),
    //             },
    //         );
    //         let node_id = builder.add_input(node).unwrap();
    //
    //         if let Some(old_id) = builder.insert_name_to_id(value_info.name, node_id) {
    //             // Todo: Rename name.
    //             println!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
    //         }
    //     }
    //
    //     for mut tensor in graph_schema.initializer {
    //         let elem_ty = tensor.data_type.try_into().unwrap();
    //         let name = tensor.name.take().ok_or(Error::MissingInputNode).unwrap();
    //
    //         let node = Node::new(
    //             Op::Const,
    //             Definition {
    //                 // Todo: we need to read the dimensions.
    //                 shape: tensor.dims.clone(),
    //                 dtype: elem_ty,
    //                 node: None,
    //                 name: tensor.name.clone().unwrap_or("".to_string()),
    //             },
    //         );
    //
    //         let node_id = builder.add_node(node).expect("TODO");
    //         builder.add_initial_tensor(node_id, tensor);
    //
    //         if let Some(old_id) = builder.insert_name_to_id(name, node_id) {
    //             // Todo: Rename name.
    //             println!("found two inputs with the same for id: prev:[{old_id}] new:[{node_id}]");
    //         }
    //     }
    //
    //     for value_info in graph_schema.output {
    //         let ty = value_info.ty.ok_or(Error::Unknown)?;
    //         // Todo: Should we support other types such as maps, sparse tensors, etc.
    //         let (elem_ty, shape) = ty.get_tensor_info().ok_or(Error::Unknown)?;
    //
    //         let mut shape_ = Vec::new();
    //         if let Some(s) = shape {
    //             shape_.extend(s);
    //         }
    //
    //         let node = Node::new(
    //             Op::NoOp,
    //             Definition {
    //                 shape: shape_,
    //                 dtype: elem_ty,
    //                 node: None,
    //                 name: value_info.name.clone(),
    //             },
    //         );
    //         let node_id = builder.add_output_node(node).expect("TODO");
    //
    //         if let Some(old_id) = builder.insert_name_to_id(value_info.name, node_id) {
    //             // Todo: Rename name.
    //             println!("found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]");
    //         }
    //     }
    //
    //     for ir_node in graph_schema.node {
    //         let op = ir_node
    //             .op_type
    //             .as_ref()
    //             .map(|op| op.parse::<Op>())
    //             .ok_or(Error::Unknown)
    //             .unwrap()
    //             .map_err(|_| Error::Unknown)
    //             .unwrap();
    //
    //         let mut node = Node::new(
    //             op,
    //             Definition {
    //                 shape: Vec::new(),
    //                 dtype: DataType::Undefined,
    //                 node: Some(ir_node.clone()),
    //                 name: ir_node.name.clone().unwrap_or("".to_string()),
    //             },
    //         );
    //
    //         // Add attributes.
    //         for attr in ir_node.attribute {
    //             // Todo: Let's avoid the copy.
    //             // Maybe we can define some type of object that we agree to never drop
    //             // and then we can leak the string.
    //             node.add_attr(attr.name.clone().into_boxed_str(), attr);
    //         }
    //
    //         for name in ir_node.input.iter() {
    //             // Todo: Mapping one name to a single node id, we lose information,
    //             // because a single node might have two outputs, how do we differentiate?
    //             let input_node_id = builder
    //                 .get_node_id(name)
    //                 .ok_or_else(|| Error::MissingInputNode)?;
    //             node.add_input(input_node_id);
    //         }
    //
    //         let node_id = builder.add_node(node).expect("TODO");
    //
    //         let mut output_node_ids = Vec::new();
    //         for name in ir_node.output {
    //             match builder.get_node_id(&name) {
    //                 None => {
    //                     let mut output_node = Node::new(
    //                         Op::NoOp,
    //                         Definition {
    //                             name: name.clone(),
    //                             ..Default::default()
    //                         },
    //                     );
    //                     output_node.add_input(node_id);
    //
    //                     let output_node_id =
    //                         builder.add_node(output_node).map_err(|_| Error::Unknown)?;
    //
    //                     output_node_ids.push(output_node_id);
    //                     builder.insert_name_to_id(name, output_node_id);
    //                 }
    //                 Some(id) => {
    //                     let node = builder.get_node_mut(id).ok_or(Error::MissingInputNode)?;
    //                     node.add_input(node_id);
    //                     output_node_ids.push(id);
    //                 }
    //             }
    //         }
    //
    //         let node = builder
    //             .get_node_mut(node_id)
    //             .expect("We just inserted it above.");
    //         for id in output_node_ids {
    //             node.add_output(id);
    //         }
    //     }
    //
    //     // for value_info in graph_schema.output {
    //     //     // let ty = value_info.ty.ok_or(Error::Unknown)?;
    //     //     // // Todo: Should we support other types such as maps, sparse tensors, etc.
    //     //     // let (elem_ty, shape) = ty.get_tensor_info().ok_or(Error::Unknown)?;
    //     //     //
    //     //     // // Todo: Add allocator API to schema.
    //     //     // let mut shape_ = Vec::new();
    //     //     // if let Some(s) = shape {
    //     //     //     shape_.extend(s);
    //     //     // }
    //     //     //
    //     //     // let node = Node::new(
    //     //     //     Op::NoOp,
    //     //     //     Definition {
    //     //     //         shape: shape_,
    //     //     //         dtype: elem_ty,
    //     //     //     },
    //     //     // );
    //     //     // let node_id = builder.add_output(node).expect("TODO");
    //     //     //
    //     //     // if let Some(old_id) = builder.insert_name_to_id(value_info.name, node_id) {
    //     //     //     // Todo: Rename name.
    //     //     //     warn!("found two outputs with the same for id: prev:[{old_id}] new:[{node_id}]");
    //     //     // }
    //     //     match builder.get_node_id(&value_info.name) {
    //     //         None => {
    //     //             todo!()
    //     //         }
    //     //         Some(node_id) => {
    //     //             let _ = builder.add_output(node_id);
    //     //         }
    //     //     }
    //     // }
    //
    //     builder.build().map_err(|_| Error::Unknown)
    todo!()
}
