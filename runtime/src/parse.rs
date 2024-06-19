use log::warn;
use rmlk_graph::device::CpuDevice;
use rmlk_graph::{GraphBuilder, Node, Op};

type Result<T> = std::result::Result<T, Error>;

pub enum Error {
    Unknown,
}

pub fn parse_graph(hir_graph: rmlk_hir::Graph) -> Result<()> {
    let device = CpuDevice;
    let mut builder = GraphBuilder::new(device.clone());

    for hir_input in hir_graph.input {
        let node = Node::new(Op::NoOp, device.clone());
        let node_id = builder.add_input(node).expect("TODO");
        if builder.store_id_by_name(hir_input.name, node_id).is_some() {
            warn!("found two inputs with the same for id {id}");
        }
    }

    for hir_output in hir_graph.output {
        let node = Node::new(Op::NoOp, device.clone());
        let node_id = builder.add_output(node).expect("TODO");
        if let Some(name) = builder.store_id_by_name(hir_output.name, node_id) {
            warn!("found two outputs with the same for id {id}");
        }
    }

    for mut hir_node in hir_graph.node {
        let op = hir_node.op_type.map(|op| op.parse::<Op>())??;
        let node = Node::new(op, device.clone());
        let node_id = builder.add_node(node).expect("TODO");
        let name = hir_node.name.take().unwrap_or(format!("node-{id}"));

        if builder.store_id_by_name(name, node_id).is_some() {
            return Err(Error::Unknown);
        }
    }

    let graph = builder.build();

    Ok(())
}
