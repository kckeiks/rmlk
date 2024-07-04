use crate::cuda::data::Data;
use crate::cuda::kernels::{add, mul};
use crate::cuda::ops;
use crate::kernel::Context;
use crate::kernel::Kernel;
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
    type Data = Data;

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
            Op::Conv => {}
            _ => todo!(),
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::cuda::data::Data;
    use crate::cuda::kernel::CudaKernel;
    use crate::execution_state::ExecutionState;
    use crate::kernel::{Context, Kernel};
    use crate::tensor::Tensor;
    use cudarc::driver::CudaDevice;
    use half::f16;
    use rmlk_graph::{Definition, Graph, GraphBuilder, Node};
    use rmlk_ir::{DataType, Op};
    use std::sync::Arc;

    fn build_binary_op_graph_and_state(
        device: Arc<CudaDevice>,
        op: Op,
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    ) -> (Arc<Graph>, ExecutionState<Data>) {
        let mut builder = GraphBuilder::new();

        let node_a = Node::new(Op::NoOp, Definition::default());
        let node_b = Node::new(Op::NoOp, Definition::default());
        // Todo: Shape of node_c probably should be precomputed when validating the graph.
        let mut node_c = Node::new(op, Definition::default());

        let a = builder.add_input(node_a).unwrap();
        let b = builder.add_input(node_b).unwrap();

        node_c.add_input(a);
        node_c.add_input(b);

        builder.add_node(node_c).unwrap();

        let graph = Arc::new(builder.build().unwrap());

        let mut tensor_a = Tensor::new(DataType::Float16, shape_a.clone());
        tensor_a.init(Data::F16(
            device
                .htod_copy(vec![
                    f16::from_f32(1.0),
                    f16::from_f32(2.0),
                    f16::from_f32(3.0),
                    f16::from_f32(4.0),
                ])
                .unwrap(),
        ));
        let mut tensor_b = Tensor::new(DataType::Float16, shape_b);
        tensor_b.init(Data::F16(
            device
                .htod_copy(vec![
                    f16::from_f32(1.0),
                    f16::from_f32(2.0),
                    f16::from_f32(3.0),
                    f16::from_f32(4.0),
                ])
                .unwrap(),
        ));
        let tensor_c = Tensor::new(DataType::Float16, shape_a);
        let state = ExecutionState::new(
            graph.clone(),
            vec![tensor_a, tensor_b, tensor_c],
            vec![0, 0, 0],
        );
        (graph, state)
    }

    #[test]
    fn test_add_f16() {
        let device = CudaDevice::new(0).unwrap();

        let shape = vec![4, 1, 1, 1];
        let (graph, state) =
            build_binary_op_graph_and_state(device.clone(), Op::Add, shape.clone(), shape);
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
    //
    // #[test]
    // fn test_add_f32() {
    //     let device = CudaDevice::new(0).unwrap();
    //     let cuda = CudaProvider::new(device);
    //
    //     let shape = [4, 1, 1, 1];
    //
    //     let mut lhs_strides = [0usize; 4];
    //     lhs_strides[0] = 1;
    //     for i in 1..4 {
    //         lhs_strides[i] += lhs_strides[i - 1] * shape[i - 1];
    //     }
    //     let mut rhs_strides = [0usize; 4];
    //     rhs_strides[0] = 1;
    //     for i in 1..4 {
    //         rhs_strides[i] += rhs_strides[i - 1] * shape[i - 1];
    //     }
    //
    //     let mut lhs_tensor = Tensor::new(DataType::Float, shape.to_vec(), lhs_strides.to_vec());
    //     let mut rhs_tensor = Tensor::new(DataType::Float, shape.to_vec(), rhs_strides.to_vec());
    //     let mut out_tensor = Tensor::new(DataType::Float, shape.to_vec(), lhs_strides.to_vec());
    //
    //     let lhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
    //     let rhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
    //
    //     lhs_tensor.init(lhs_data);
    //     rhs_tensor.init(rhs_data);
    //
    //     cuda.forward(Op::Add, &lhs_tensor, &rhs_tensor, &mut out_tensor)
    //         .unwrap();
    //     let result = cuda.dtoh_f32(out_tensor.data().unwrap()).unwrap();
    //     assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    // }
    //
    // #[test]
    // fn test_mul_f16() {
    //     let device = CudaDevice::new(0).unwrap();
    //     let cuda = CudaProvider::new(device);
    //
    //     let shape = [3, 4, 5];
    //     let elem_num = shape.iter().product::<usize>();
    //     let dims = shape.len();
    //
    //     let mut lhs_strides = vec![0usize; dims];
    //     lhs_strides[0] = 1;
    //     for i in 1..dims {
    //         lhs_strides[i] += lhs_strides[i - 1] * shape[i - 1];
    //     }
    //     let mut rhs_strides = vec![0usize; dims];
    //     rhs_strides[0] = 1;
    //     for i in 1..dims {
    //         rhs_strides[i] += rhs_strides[i - 1] * shape[i - 1];
    //     }
    //
    //     let mut lhs_tensor = Tensor::new(DataType::Float16, shape.to_vec(), lhs_strides.clone());
    //     let mut rhs_tensor = Tensor::new(DataType::Float16, shape.to_vec(), rhs_strides);
    //     let mut out_tensor = Tensor::new(DataType::Float16, shape.to_vec(), lhs_strides);
    //
    //     let lhs_data = cuda.htod_f16(vec![f16::from_f32(2.0); elem_num]).unwrap();
    //     let rhs_data = cuda.htod_f16(vec![f16::from_f32(3.0); elem_num]).unwrap();
    //
    //     lhs_tensor.init(lhs_data);
    //     rhs_tensor.init(rhs_data);
    //
    //     cuda.forward(Op::Mul, &lhs_tensor, &rhs_tensor, &mut out_tensor)
    //         .unwrap();
    //     let result = cuda.dtoh_f16(out_tensor.data().unwrap()).unwrap();
    //     assert_eq!(result, vec![f16::from_f32(6.0); elem_num])
    // }
    //
    // #[test]
    // fn test_matmul_f32() {
    //     let device = CudaDevice::new(0).unwrap();
    //     let cuda = CudaProvider::new(device);
    //
    //     let shape = [1, 2, 2];
    //
    //     let mut lhs_strides = [0usize; 3];
    //     lhs_strides[2] = 1;
    //     for i in (0..2).rev() {
    //         lhs_strides[i] += lhs_strides[i + 1] * shape[i + 1];
    //     }
    //     let mut rhs_strides = [0usize; 3];
    //     rhs_strides[2] = 1;
    //     for i in (0..2).rev() {
    //         rhs_strides[i] += rhs_strides[i + 1] * shape[i + 1];
    //     }
    //
    //     let mut lhs_tensor = Tensor::new(DataType::Float, shape.to_vec(), lhs_strides.to_vec());
    //     let mut rhs_tensor = Tensor::new(DataType::Float, shape.to_vec(), rhs_strides.to_vec());
    //     let mut out_tensor = Tensor::new(DataType::Float, shape.to_vec(), lhs_strides.to_vec());
    //
    //     let lhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
    //     let rhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
    //
    //     lhs_tensor.init(lhs_data);
    //     rhs_tensor.init(rhs_data);
    //
    //     cuda.matmul(&lhs_tensor, &rhs_tensor, &mut out_tensor)
    //         .unwrap();
    //     let result = cuda.dtoh_f32(out_tensor.data().unwrap()).unwrap();
    //     assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    // }
}
