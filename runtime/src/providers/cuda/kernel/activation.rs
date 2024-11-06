use crate::core::Context;
use crate::core::{Error, Result};
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::driver::CudaDevice;
use log::trace;
use rmlk_schema::DataType;
use std::sync::Arc;

pub struct ActivationKernel {
    device: Arc<CudaDevice>,
}

impl ActivationKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        trace!("x_shape={x_shape:?},x_stride={x_stride:?}");

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x.data().and_then(|data| data.f32()).ok_or_else(|| {
                Error::Internal("expected tensor data to be of type `float32`".to_string())
            })?;

            let mut y_data = self.device.alloc_zeros(x.shape().iter().product())?;

            rmlk_cuda::kernels::activation::compute(
                self.device,
                (1.0, 0.0),
                x_data,
                &x_shape,
                &x_stride,
                &mut y_data,
            )?;

            let output_shape = x.shape().clone();
            let output = ctx.get_output_mut(0)?;
            output.init(CudaData::F32(y_data));
            output._reshape(output_shape);
            output.set_dtype(DataType::Float);
        } else {
            return Err(Error::NoSupport(format!(
                "unsupported dtype `{:?}`",
                x.dtype()
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::activation::ActivationKernel;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

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

        let params = TestParams {
            inputs: vec![node_a],
            attributes: Vec::new(),
            op: Op::Relu,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 2).unwrap();

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
