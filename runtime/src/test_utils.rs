use crate::core::Values;
use crate::core::{Context, Kernel, ModelInstanceState};
use crate::core::{DeviceService, ExecutionState, Plan};
use rmlk_graph::{GraphBuilder, Node};
use rmlk_schema::{Attribute, AttributeType, DataType, Definition, Op};
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
    pub op: Op,
    pub attributes: Vec<Attribute>,
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
        unimplemented!()
    }

    fn htod_float(&self, _: Vec<f32>) -> crate::Result<Self::Data> {
        unimplemented!()
    }

    fn dtoh_float(&self, _: &mut Self::Data) -> crate::Result<Vec<f32>> {
        unimplemented!()
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

    let mut op_node = Node::from_definition(params.op, Definition::default());

    for attr in params.attributes {
        op_node.add_attr(attr.name.clone().into_boxed_str(), attr);
    }

    let mut inputs = Vec::new();
    for (index, input) in params.inputs.into_iter().enumerate() {
        let input_node = Node::from_definition(
            Op::NoOp,
            Definition {
                shape: input.shape.clone(),
                dtype: input.dtype,
                node: None,
                name: format!("{index}-input"),
            },
        );
        let node_id = builder.add_input(input_node).unwrap();
        op_node.add_input(node_id);

        inputs.push((node_id, input.data.unwrap()));
    }

    let output_node = Node::from_definition(Op::NoOp, Definition::default());
    let output_id = builder.add_node(output_node).unwrap();
    op_node.add_output(output_id);

    builder.add_node(op_node).unwrap();
    let graph = builder.build().unwrap();

    let mut values = Values::new(&provider, &graph).unwrap();
    for (node_id, data) in inputs {
        let tensor = values.get_mut(node_id).unwrap();
        tensor.init(data);
    }

    let instance_state = ModelInstanceState::new(Plan::new(Box::new([provider])), graph);

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
