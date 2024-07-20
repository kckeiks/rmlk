use crate::cuda::data::CudaData;
use crate::cuda::kernels::{add, mul};
use crate::cuda::ops;
use crate::kernel::Context;
use crate::kernel::Kernel;
use crate::{op, Error, Result};
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
            Op::MaxPool => {
                ops::max_pool::compute(ctx, self.device.clone())?;
            }
            Op::GlobalAveragePool => {
                ops::global_average_pool::compute(ctx, self.device.clone())?;
            }
            Op::Relu => {
                ops::activation::compute(ctx, self.device.clone())?;
            }
            Op::Flatten => {
                op::flatten::compute::<CudaData>(ctx)?;
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
    use crate::kernel::{Context, Kernel};
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

    // Todo: Move this to `op` once we have a mock provider.
    #[test]
    fn test_flatten_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 4, 4];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![
                        1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0,
                        4.0,
                    ])
                    .unwrap(),
            )),
        };

        let node_c = TestNode {
            shape: vec![],
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes: Vec::new(),
            op: Op::Flatten,
        };

        let (_, state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(state, 1).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Flatten, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let shape = context.get_output(0).unwrap().shape();

        assert_eq!(shape, &vec![1, 16])
    }
}
