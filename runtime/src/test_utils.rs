use crate::core::{Context, Kernel, ModelInstanceState};
use crate::core::{DeviceService, ExecutionState, Plan};
use crate::core::{Tensor, Values};
use rmlk_graph::{Definition, GraphBuilder, Node};
use rmlk_ir::{Attribute, AttributeType, DataType, Op};
use std::collections::HashMap;
use std::marker::PhantomData;
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

pub struct MockProvider<T> {
    _marker: PhantomData<T>,
}

impl<T> MockProvider<T> {
    pub fn new() -> Self {
        MockProvider {
            _marker: PhantomData,
        }
    }
}

impl<T> DeviceService for MockProvider<T> {
    type Data = Vec<T>;
    type Kernel = MockKernel<T>;

    fn get_kernel(&self, _: Op, _: DataType) -> crate::Result<Self::Kernel> {
        todo!()
    }

    fn htod_float(&self, _: Vec<f32>) -> crate::Result<Self::Data> {
        todo!()
    }

    fn dtoh_float(&self, _: &mut Self::Data) -> crate::Result<Vec<f32>> {
        todo!()
    }
}

pub struct MockKernel<T> {
    _marker: PhantomData<T>,
}

impl<T> Kernel for MockKernel<T> {
    type Device = MockProvider<T>;

    fn compute(self, _: &mut Context<Self::Device>) -> crate::Result<()> {
        todo!()
    }
}

pub fn build_graph_and_state<T, P: DeviceService<Data = T>>(
    provider: P,
    params: TestParams<T>,
) -> ExecutionState<P> {
    let mut builder = GraphBuilder::new();

    let mut out_node = Node::new(params.op, Definition::default());

    let mut all_tensors = Vec::new();

    // Mapping node id to its index in the tensors buffer.
    let mut node_to_tensor_index = HashMap::new();

    // Mapping from a node to the indices of all of its inputs and outputs.
    // Note: the key here is not the node id.
    let mut node_tensors = Vec::new();

    for attr in params.attributes {
        out_node.add_attr(attr.name.clone().into_boxed_str(), attr);
    }

    let mut inputs = Vec::new();
    for (index, input) in params.inputs.into_iter().enumerate() {
        let input_node = Node::new(
            Op::NoOp,
            Definition {
                shape: input.shape.clone(),
                dtype: input.dtype,
                node: None,
                name: format!("{index}-input"),
            },
        );
        let node_id = builder.add_input(input_node).unwrap();
        out_node.add_input(node_id);

        let tensor: Tensor<T> = Tensor::new_with_shape(input.dtype, input.shape.clone());
        // tensor.init(input.data.clone().unwrap());
        inputs.push((node_id, input.data.unwrap()));

        let current_index = all_tensors.len();
        all_tensors.push(Some(tensor));
        node_tensors.push(current_index);
        node_to_tensor_index.insert(node_id, current_index);
    }

    for input in out_node.inputs() {
        let tensor_index = node_to_tensor_index.get(input).unwrap();
        node_tensors.push(*tensor_index);
    }

    let output = Node::new(Op::NoOp, Definition::default());
    let output_id = builder.add_node(output).unwrap();

    out_node.add_output(output_id);

    builder.add_node(out_node).unwrap();

    let graph = builder.build().unwrap();

    let out_tensor = if params.outputs[0].shape.is_empty() {
        Tensor::new(params.outputs[0].dtype)
    } else {
        Tensor::new_with_shape(params.outputs[0].dtype, params.outputs[0].shape.clone())
    };

    let current_index = all_tensors.len();
    all_tensors.push(Some(out_tensor));
    node_tensors.push(current_index);

    let mut values = Values::new(&provider, &graph).unwrap();

    for (node_id, data) in inputs {
        // println!("node_id={node_id}");
        let tt = values.get_mut(node_id).unwrap();
        tt.init(data);
    }

    // Todo: finish.
    let instance_state = ModelInstanceState::new(
        // Todo: finish.
        Plan::new(Box::new([provider])),
        graph,
    );

    ExecutionState::new(Arc::new(instance_state), values).unwrap()
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
