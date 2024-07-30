use crate::core::context::Context;
use crate::core::error::{Error, Result};
use crate::core::provider::cuda::data::CudaData;
use cudarc::driver::CudaDevice;
use rmlk_ir::DataType;
use std::sync::Arc;

pub struct GlobalAveragePoolKernel {
    device: Arc<CudaDevice>,
}

impl GlobalAveragePoolKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    pub fn compute(self, ctx: &mut Context<CudaData>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let mut y_shape = vec![0; x_shape.len()];
        rmlk_cuda::cuda::global_average_pool::compute_output_shape(
            &x_shape,
            y_shape.as_mut_slice(),
        )
        .unwrap();

        let mut y_stride = vec![0; y_shape.len()];
        rmlk_cuda::calculate_stride(&y_shape, &mut y_stride);

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;

            let mut y_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
                .unwrap();

            rmlk_cuda::cuda::global_average_pool::compute::<f32>(
                self.device,
                (1.0, 0.0),
                &x_data,
                &x_shape,
                &x_stride,
                &mut y_data,
                &y_shape,
                &y_stride,
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
    use crate::core::provider::cuda::data::CudaData;
    use crate::core::provider::cuda::kernel::global_average_pool::GlobalAveragePoolKernel;
    use crate::core::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

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

        let node_c = TestNode {
            shape: vec![1, 1, 1, 1],
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes: Vec::new(),
            op: Op::GlobalAveragePool,
        };

        let mut state = crate::core::test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 1).unwrap();

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
