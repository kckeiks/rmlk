use crate::core::device_service::{DeviceService, DeviceServiceError};
use crate::core::kernel::{KernelError, Result};
use crate::core::Context;
use crate::ops::global_average::GlobalAverage;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::driver::CudaDevice;
use rmlk_schema::DataType;
use std::sync::Arc;

pub struct GlobalAveragePoolKernel {
    device: Arc<CudaDevice>,
}

impl GlobalAveragePoolKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    // pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
    //     let x = ctx.get_input(0)?;
    //     let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
    //     let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
    //
    //     let mut y_shape = vec![0; x_shape.len()];
    //     rmlk_cuda::kernels::global_average_pool::compute_output_shape(
    //         &x_shape,
    //         y_shape.as_mut_slice(),
    //     )?;
    //
    //     let mut y_stride = vec![0; y_shape.len()];
    //     utils::calculate_stride(&y_shape, &mut y_stride);
    //
    //     if matches!(x.dtype(), DataType::Float) {
    //         let x_data = x.data().and_then(|data| data.f32()).ok_or_else(|| {
    //             KernelError::Other("expected tensor data to be of type `float32`".to_string())
    //         })?;
    //
    //         let mut y_data = self
    //             .device
    //             .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
    //             .map_err(rmlk_cuda::Error::from)
    //             .map_err(DeviceServiceError::from)?;
    //
    //         rmlk_cuda::kernels::global_average_pool::compute::<f32>(
    //             self.device,
    //             (1.0, 0.0),
    //             &x_data,
    //             &x_shape,
    //             &x_stride,
    //             &mut y_data,
    //             &y_shape,
    //             &y_stride,
    //         )?;
    //
    //         let output = ctx.get_output_mut(0)?;
    //         output.init(CudaData::F32(y_data));
    //         output._reshape(y_shape.iter().map(|d| *d as usize).collect());
    //         output.set_dtype(DataType::Float);
    //     } else {
    //         return Err(KernelError::Other(format!(
    //             "unsupported dtype `{:?}`",
    //             x.dtype()
    //         )));
    //     }
    //
    //     Ok(())
    // }
}

impl GlobalAverage for GlobalAveragePoolKernel {
    type Service = Cuda;

    fn compute(
        self,
        x_data: &<Self::Service as DeviceService>::Data,
        x_shape: &[i32],
        x_stride: &[i32],
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<<Self::Service as DeviceService>::Data> {
        if x_data.f32().is_some() {
            let x_data = x_data.f32().ok_or_else(|| {
                KernelError::Other("expected tensor data to be of type `float32`".to_string())
            })?;

            let mut y_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
                .map_err(rmlk_cuda::Error::from)
                .map_err(DeviceServiceError::from)?;

            rmlk_cuda::kernels::global_average_pool::compute::<f32>(
                self.device,
                (1.0, 0.0),
                &x_data,
                &x_shape,
                &x_stride,
                &mut y_data,
                &y_shape,
                &y_stride,
            )?;

            Ok(CudaData::F32(y_data))
        } else {
            return Err(KernelError::Other(
                "unsupported dtype for global average".to_string(),
            ));
        }
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::global_average_pool::GlobalAveragePoolKernel;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_global_average_pool_f32_2d() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 3, 3];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                    .unwrap(),
            )),
        };

        let params = TestParams {
            inputs: vec![node_a],
            attributes: vec![],
            op: Op::GlobalAveragePool,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let cuda_kernel = GlobalAveragePoolKernel::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![5.0])
    }
}
