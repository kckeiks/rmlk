use crate::core::device_service::{DeviceService, DeviceServiceError};
use crate::core::kernel::{KernelError, Result};
use crate::core::Tensor;
use crate::ops::max_pool::MaxPool;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::driver::CudaDevice;
use rmlk_schema::DataType;
use std::sync::Arc;

pub struct MaxPoolKernel {
    device: Arc<CudaDevice>,
}

impl MaxPoolKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

impl MaxPool for MaxPoolKernel {
    type Service = Cuda;

    fn compute(
        self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        kernel_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<<Self::Service as DeviceService>::Data> {
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x.data().and_then(|data| data.f32()).ok_or_else(|| {
                KernelError::Other("expected tensor data to be of type `float32`".to_string())
            })?;

            // Todo: move this to DeviceService trait.
            let mut y_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
                .map_err(rmlk_cuda::Error::from)
                .map_err(DeviceServiceError::from)?;

            rmlk_cuda::kernels::max_pool::compute::<f32>(
                self.device,
                (1.0, 0.0),
                &x_data,
                &x_shape,
                &x_stride,
                kernel_shape,
                pads,
                strides,
                &mut y_data,
                &y_shape,
                &y_stride,
            )?;

            Ok(CudaData::F32(y_data))
        } else {
            todo!()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::max_pool::MaxPoolKernel;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestMaxPoolAttributes, TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_max_pool_f32_2d() {
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

        let attributes = test_utils::create_max_pool_attributes(TestMaxPoolAttributes {
            dilations: None,
            kernel_shape: Some(Box::new([2, 2])),
            strides: Some(Box::new([2, 2])),
            row_major_order: None,
            ceil_mode: None,
            pads: None,
        });

        let params = TestParams {
            inputs: vec![node_a],
            attributes,
            op: Op::MaxPool,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let cuda_kernel = MaxPoolKernel::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 3.0, 4.0])
    }
}
