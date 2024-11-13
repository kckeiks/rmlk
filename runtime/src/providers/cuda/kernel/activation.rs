use crate::core::device_service::{DeviceService, DeviceServiceError};
use crate::core::kernel::{KernelError, Result};
use crate::ops::activation::Activation;
use crate::providers::cuda::data::CudaData;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

pub struct ActivationKernel {
    device: Arc<CudaDevice>,
}

impl ActivationKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

impl Activation for ActivationKernel {
    type Data = CudaData;
    fn compute(
        &self,
        x_data: &Self::Data,
        x_shape: &[i32],
        x_stride: &[i32],
    ) -> Result<Self::Data> {
        match x_data {
            CudaData::F32(x_data) => {
                let mut y_data = self
                    .device
                    .alloc_zeros::<f32>(x_shape.iter().product())
                    .map_err(rmlk_cuda::Error::from)
                    .map_err(DeviceServiceError::from)?;

                rmlk_cuda::kernels::activation::compute::<f32>(
                    self.device.clone(),
                    (1.0, 0.0),
                    x_data,
                    x_shape,
                    x_stride,
                    &mut y_data,
                )?;

                Ok(CudaData::F32(y_data))
            }
            _ => Err(KernelError::Other("".to_string())),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::activation::ActivationOp;
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

        let cuda_kernel = ActivationOp::new(device.clone());
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
