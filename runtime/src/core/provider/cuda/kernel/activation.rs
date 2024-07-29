use crate::core::context::Context;
use crate::core::error::{Error, Result};
use cudarc::driver::CudaDevice;
use rmlk_ir::DataType;
use rmlk_tensor::cuda::CudaData;
use std::sync::Arc;

pub struct ActivationKernel {
    device: Arc<CudaDevice>,
}

impl ActivationKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    pub fn compute(self, ctx: &mut Context<CudaData>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;

            let mut y_data = self
                .device
                .alloc_zeros(x.shape().iter().product())
                .map_err(|_| Error::MissingData)?;

            rmlk_tensor::cuda::activation::compute(
                self.device,
                (1.0, 0.0),
                x_data,
                &x_shape,
                &x_stride,
                &mut y_data,
            )
            .map_err(|_| Error::ComputationFailed)?;

            let output = ctx.get_output_mut(0)?;
            output.init(CudaData::F32(y_data));
        } else {
            todo!()
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::context::Context;
    use crate::core::provider::cuda::kernel::activation::ActivationKernel;
    use crate::core::provider::cuda::test_utils;
    use crate::core::provider::cuda::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};
    use rmlk_tensor::cuda::CudaData;

    #[test]
    fn test_relu_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 2, 2];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![-1.0, 2.0, -3.0, 100.0]).unwrap(),
            )),
        };

        let node_c = TestNode {
            shape: vec![1, 1, 2, 2],
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes: Vec::new(),
            op: Op::Relu,
        };

        let mut state = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 1).unwrap();

        let cuda_kernel = ActivationKernel::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![0.0, 2.0, 0.0, 100.0])
    }
}
