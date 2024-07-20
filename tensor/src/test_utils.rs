use crate::{ExecutionState, Tensor};
use rmlk_graph::{Definition, Graph, GraphBuilder, Node};
use rmlk_ir::{Attribute, AttributeType, DataType, Op};
use std::sync::Arc;

pub struct TestConvAttributes {
    pub dilations: Option<Box<[i32]>>,
    pub group: Option<i32>,
    pub kernel_shape: Option<Box<[i32]>>,
    pub pads: Option<Box<[i32]>>,
    pub strides: Option<Box<[i32]>>,
}

pub struct TestMaxPoolAttributes {
    pub dilations: Option<Box<[i32]>>,
    pub ceil_mode: Option<i32>,
    pub kernel_shape: Option<Box<[i32]>>,
    pub pads: Option<Box<[i32]>>,
    pub strides: Option<Box<[i32]>>,
    pub row_major_order: Option<i32>,
}

pub struct TestNode<T> {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub data: Option<T>,
}

pub struct TestParams<T> {
    pub inputs: Vec<TestNode<T>>,
    pub outputs: Vec<TestNode<T>>,
    pub attributes: Vec<Attribute>,
    pub op: Op,
}

pub fn build_graph_and_state<T>(params: TestParams<T>) -> (Arc<Graph>, ExecutionState<T>) {
    let mut builder = GraphBuilder::new();

    let mut out_node = Node::new(params.op, Definition::default());

    let mut tensors = Vec::new();

    for attr in params.attributes {
        out_node.add_attr(attr.name.clone().into_boxed_str(), attr);
    }

    for input in params.inputs {
        let input_node = Node::new(Op::NoOp, Definition::default());
        let node_id = builder.add_input(input_node).unwrap();
        out_node.add_input(node_id);

        let mut tensor = Tensor::new_with_shape(input.dtype, input.shape.clone());
        tensor.init(input.data.unwrap());
        tensors.push(tensor)
    }

    builder.add_node(out_node).unwrap();

    let graph = Arc::new(builder.build().unwrap());

    let out_tensor = if params.outputs[0].shape.is_empty() {
        Tensor::new(params.outputs[0].dtype)
    } else {
        Tensor::new_with_shape(params.outputs[0].dtype, params.outputs[0].shape.clone())
    };

    tensors.push(out_tensor);

    let state = ExecutionState::new(
        graph.clone(),
        tensors,
        // Todo: Update.
        vec![0, 0, 0, 0],
    );
    (graph, state)
}

pub fn create_conv_attributes(conv_attrs: TestConvAttributes) -> Vec<Attribute> {
    let mut result = Vec::new();
    result.push(Attribute {
        name: "dilations".to_string(),
        ref_attr_name: None,
        doc_string: None,
        ty: AttributeType::Ints(conv_attrs.dilations.unwrap().to_vec()),
    });

    result.push(Attribute {
        name: "group".to_string(),
        ref_attr_name: None,
        doc_string: None,
        ty: AttributeType::Int(conv_attrs.group.unwrap()),
    });

    result.push(Attribute {
        name: "pads".to_string(),
        ref_attr_name: None,
        doc_string: None,
        ty: AttributeType::Ints(conv_attrs.pads.unwrap().to_vec()),
    });

    if conv_attrs.kernel_shape.is_some() {
        result.push(Attribute {
            name: "kernel_shape".to_string(),
            ref_attr_name: None,
            doc_string: None,
            ty: AttributeType::Ints(conv_attrs.kernel_shape.unwrap().to_vec()),
        });
    }

    result.push(Attribute {
        name: "strides".to_string(),
        ref_attr_name: None,
        doc_string: None,
        ty: AttributeType::Ints(conv_attrs.strides.unwrap().to_vec()),
    });

    result
}

pub fn create_max_pool_attributes(pool_attrs: TestMaxPoolAttributes) -> Vec<Attribute> {
    let mut result = Vec::new();

    if pool_attrs.dilations.is_some() {
        result.push(Attribute {
            name: "dilations".to_string(),
            ref_attr_name: None,
            doc_string: None,
            ty: AttributeType::Ints(pool_attrs.dilations.unwrap().to_vec()),
        });
    }

    if pool_attrs.ceil_mode.is_some() {
        result.push(Attribute {
            name: "ceil_mode".to_string(),
            ref_attr_name: None,
            doc_string: None,
            ty: AttributeType::Int(pool_attrs.ceil_mode.unwrap()),
        });
    }

    if pool_attrs.row_major_order.is_some() {
        result.push(Attribute {
            name: "row_major_order".to_string(),
            ref_attr_name: None,
            doc_string: None,
            ty: AttributeType::Int(pool_attrs.row_major_order.unwrap()),
        });
    }

    if pool_attrs.pads.is_some() {
        result.push(Attribute {
            name: "pads".to_string(),
            ref_attr_name: None,
            doc_string: None,
            ty: AttributeType::Ints(pool_attrs.pads.unwrap().to_vec()),
        });
    }

    if pool_attrs.kernel_shape.is_some() {
        result.push(Attribute {
            name: "kernel_shape".to_string(),
            ref_attr_name: None,
            doc_string: None,
            ty: AttributeType::Ints(pool_attrs.kernel_shape.unwrap().to_vec()),
        });
    }

    result.push(Attribute {
        name: "strides".to_string(),
        ref_attr_name: None,
        doc_string: None,
        ty: AttributeType::Ints(pool_attrs.strides.unwrap().to_vec()),
    });

    result
}
