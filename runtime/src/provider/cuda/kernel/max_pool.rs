use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::Context;
use crate::core::{Error, Result};
use crate::provider::cuda::data::CudaData;
use crate::utils;
use cudarc::driver::CudaDevice;
use log::trace;
use rmlk_ir::DataType;
use std::sync::Arc;

pub struct MaxPoolKernel {
    device: Arc<CudaDevice>,
}

impl MaxPoolKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    pub fn compute(self, ctx: &mut Context<CudaData>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let attrs = MaxPoolAttributes::new(ctx.get_attributes().ok_or(Error::MissingAttributes)?)?;

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        rmlk_cuda::kernels::max_pool::compute_output_shape(
            &x_shape,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &mut y_shape,
            false,
        )
        .map_err(|_| Error::ComputationFailed)?;

        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        trace!(
            "x_shape={x_shape:?},\
            x_stride={x_stride:?},\
            kernel_shape={:?},\
            pads={:?},\
            strides={:?}\
            y_shape={y_shape:?}\
            y_stride={y_stride:?}",
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides()
        );

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;

            let mut y_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())?;

            rmlk_cuda::kernels::max_pool::compute::<f32>(
                self.device,
                (1.0, 0.0),
                &x_data,
                &x_shape,
                &x_stride,
                attrs.kernel_shape(),
                attrs.pads(),
                attrs.strides(),
                &mut y_data,
                &y_shape,
                &y_stride,
            )
            .map_err(|_| Error::ComputationFailed)?;

            let output = ctx.get_output_mut(0)?;
            output.init(CudaData::F32(y_data));
            output._reshape(y_shape.iter().map(|d| *d as usize).collect());
            output.set_dtype(DataType::Float);
        } else {
            todo!()
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::provider::cuda::data::CudaData;
    use crate::provider::cuda::kernel::max_pool::MaxPoolKernel;
    use crate::test_utils;
    use crate::test_utils::{TestMaxPoolAttributes, TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};
    use std::collections::HashMap;

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

        let node_c = TestNode {
            shape: vec![1, 1, 2, 2],
            dtype,
            data: None,
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
            outputs: vec![node_c],
            attributes,
            op: Op::MaxPool,
        };

        let mut state = test_utils::build_graph_and_state(params);
        let map = HashMap::from([(1, 1)]);
        let mut context = Context::new(&mut state, &map, 1).unwrap();

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
