use crate::cuda::data::CudaData;
use crate::cuda::kernels::{add, mul};
use crate::cuda::ops;
use crate::kernel::Kernel;
use crate::kernel::{Context, ConvAttributes};
use crate::{Error, Result};
use cudarc::driver::{CudaDevice, CudaFunction};
use rmlk_ir::{DataType, Op};
use std::sync::Arc;

pub struct CudaKernel {
    op: Op,
    device: Arc<CudaDevice>,
}

impl CudaKernel {
    pub fn new(op: Op, device: Arc<CudaDevice>) -> Self {
        Self { op, device }
    }

    fn kernel(&self, dtype: DataType) -> Result<CudaFunction> {
        let (fwd_fn_name, fwd_fn_all, module_name, ptx_src) = match self.op {
            Op::Add => (
                add::FWD_FN_NAMES[dtype as usize],
                add::FWD_FN_NAMES.as_slice(),
                add::MODULE_NAME,
                add::PTX_SRC,
            ),
            Op::Mul => (
                mul::FWD_FN_NAMES[dtype as usize],
                mul::FWD_FN_NAMES.as_slice(),
                mul::MODULE_NAME,
                mul::PTX_SRC,
            ),
            _ => unimplemented!(),
        };

        if !self.device.has_func(module_name, fwd_fn_name) {
            self.device
                .load_ptx(ptx_src.into(), module_name, fwd_fn_all)
                .map_err(|_| Error::Unknown)?
        }
        Ok(self
            .device
            .get_func(module_name, fwd_fn_name)
            .expect("To have been loaded"))
    }
}

impl Kernel for CudaKernel {
    type Data = CudaData;

    fn compute(&self, ctx: &mut Context<Self::Data>) -> Result<()> {
        match self.op {
            Op::Gemm => {
                ops::gemm::compute(ctx, self.device.clone())?;
            }
            Op::MatMul => {
                ops::gemm::compute(ctx, self.device.clone())?;
            }
            Op::Add => {
                // Todo: More validation.
                let dtype = ctx.get_input(0)?.dtype();
                let func = self.kernel(*dtype)?;
                ops::add::compute(ctx, self.device.clone(), func)?;
            }
            Op::Conv => {
                ops::conv::compute(ctx, self.device.clone())?;
            }
            _ => todo!(),
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::CudaKernel;
    use crate::execution_state::ExecutionState;
    use crate::kernel::{Context, ConvAttributes, Kernel};
    use crate::tensor::Tensor;
    use cudarc::driver::CudaDevice;
    use half::f16;
    use rmlk_graph::{Definition, Graph, GraphBuilder, Node};
    use rmlk_ir::{Attribute, AttributeType, DataType, Op};
    use std::sync::Arc;

    struct TestNode {
        shape: Vec<usize>,
        dtype: DataType,
        data: Option<CudaData>,
    }

    struct TestParams {
        inputs: Vec<TestNode>,
        outputs: Vec<TestNode>,
        attributes: Vec<Attribute>,
        op: Op,
    }

    fn build_graph_and_state(params: TestParams) -> (Arc<Graph>, ExecutionState<CudaData>) {
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

            let mut tensor = Tensor::new(input.dtype, input.shape.clone());
            tensor.init(input.data.unwrap());
            tensors.push(tensor)
        }

        builder.add_node(out_node).unwrap();

        let graph = Arc::new(builder.build().unwrap());

        let out_tensor = Tensor::new(params.outputs[0].dtype, params.outputs[0].shape.clone());

        tensors.push(out_tensor);

        let state = ExecutionState::new(
            graph.clone(),
            tensors,
            // Todo: Update.
            vec![0, 0, 0],
        );
        (graph, state)
    }

    #[test]
    fn test_add_f16() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![4, 1, 1, 1];
        let dtype = DataType::Float16;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F16(
                device
                    .htod_copy(vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ])
                    .unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F16(
                device
                    .htod_copy(vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ])
                    .unwrap(),
            )),
        };

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes: vec![],
            op: Op::Add,
        };

        let (_, state) = build_graph_and_state(params);
        let mut context = Context::new(state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Add, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f16()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(
            result,
            vec![
                f16::from_f32(2.0),
                f16::from_f32(4.0),
                f16::from_f32(6.0),
                f16::from_f32(8.0),
            ]
        )
    }

    #[test]
    fn test_add_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![4, 1, 1, 1];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes: vec![],
            op: Op::Add,
        };

        let (_, state) = build_graph_and_state(params);
        let mut context = Context::new(state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Add, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }

    #[test]
    fn test_gemm_f16() {
        let device = CudaDevice::new(0).unwrap();

        let shape = vec![1, 2, 2];
        let dtype = DataType::Float16;
        let op = Op::Gemm;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F16(
                device
                    .htod_copy(vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ])
                    .unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F16(
                device
                    .htod_copy(vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ])
                    .unwrap(),
            )),
        };

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes: vec![],
            op,
        };

        let (_, state) = build_graph_and_state(params);
        let mut context = Context::new(state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(op, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f16()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(
            result,
            vec![
                f16::from_f32(7.0),
                f16::from_f32(10.0),
                f16::from_f32(15.0),
                f16::from_f32(22.0)
            ]
        )
    }

    #[test]
    fn test_gemm_f32() {
        let device = CudaDevice::new(0).unwrap();

        let shape = vec![1, 2, 2];
        let dtype = DataType::Float;
        let op = Op::Gemm;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes: vec![],
            op,
        };

        let (_, state) = build_graph_and_state(params);
        let mut context = Context::new(state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(op, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    }

    fn create_conv_attributes(conv_attrs: ConvAttributes) -> Vec<Attribute> {
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

    #[test]
    fn test_conv_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 5, 5];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                    ])
                    .unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: vec![1, 1, 3, 3],
            dtype,
            data: Some(CudaData::F32(device.htod_copy(vec![1.0; 9]).unwrap())),
        };

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let attributes = create_conv_attributes(ConvAttributes {
            dilations: Some(Box::new([1, 1])),
            group: Some(1),
            kernel_shape: None,
            pads: Some(Box::new([1, 1, 1, 1])),
            strides: Some(Box::new([1, 1])),
        });

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes,
            op: Op::Conv,
        };

        let (_, state) = build_graph_and_state(params);
        let mut context = Context::new(state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Conv, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(
            result,
            vec![
                12.0, 21.0, 27.0, 33.0, 24.0, 33.0, 54.0, 63.0, 72.0, 51.0, 63.0, 99.0, 108.0,
                117.0, 81.0, 93.0, 144.0, 153.0, 162.0, 111.0, 72.0, 111.0, 117.0, 123.0, 84.0,
            ]
        )
    }
}
