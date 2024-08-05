use crate::attributes::conv::ConvAttributes;
use crate::core::Context;
use crate::core::{Error, Result};
use crate::providers::cuda::data::CudaData;
use crate::utils;
use cudarc::driver::{CudaDevice, DeviceSlice};
use log::trace;
use rmlk_cuda::kernels::conv::BiasInput;
use rmlk_ir::DataType;
use std::sync::Arc;

pub struct ConvKernel {
    device: Arc<CudaDevice>,
}

impl ConvKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    pub fn compute(self, ctx: &mut Context<CudaData>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let w = ctx.get_input(1)?;

        let filter_dims = match x.shape().len() {
            4 => 2,
            5 => 3,
            _ => {
                unreachable!("we already checked the dimensions of x for the supported dimensions")
            }
        };

        let attrs = ConvAttributes::new(
            ctx.get_attributes().ok_or(Error::MissingAttributes)?,
            filter_dims,
        )?;

        let w_shape = w.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        rmlk_cuda::kernels::conv::calculate_output_shape(
            &x_shape,
            &w_shape,
            attrs.pads(),
            attrs.strides(),
            attrs.dilations(),
            &mut y_shape,
        )
        .map_err(|_| Error::ComputationFailed)?;

        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        if matches!(x.dtype(), DataType::Float) {
            let x_data = x
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;
            let w_data = w
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;

            let mut y_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|d| *d as usize).product())?;

            match ctx.get_input(2).ok() {
                None => {
                    trace!(
                        "x_data_len={:?},\
                        x_shape={x_shape:?},\
                        x_stride={x_stride:?},\
                        w_data_len={:?},\
                        w_shape={w_shape:?},\
                        pads={:?},\
                        strides={:?},\
                        dilations={:?},\
                        group={:?},\
                        y_data_len={:?},\
                        y_shape={y_shape:?},\
                        y_stride={y_stride:?}",
                        x_data.len(),
                        w_data.len(),
                        attrs.pads(),
                        attrs.strides(),
                        attrs.dilations(),
                        attrs.group(),
                        y_data.len(),
                    );

                    rmlk_cuda::kernels::conv::compute::<f32>(
                        self.device,
                        (1.0, 0.0),
                        &x_data,
                        &x_shape,
                        &x_stride,
                        &w_data,
                        &w_shape,
                        attrs.pads(),
                        attrs.strides(),
                        attrs.dilations(),
                        attrs.group(),
                        None,
                        &mut y_data,
                        &y_shape,
                        &y_stride,
                    )
                    .map_err(|_| Error::ComputationFailed)?;
                }
                Some(bias) => {
                    let mut bias_shape = vec![1i32; x_shape.len()];
                    // Todo: Urgent. We need to make this generic.
                    bias_shape[1] = bias.shape()[0] as i32;

                    let mut bias_stride = vec![0i32; x_shape.len()];
                    utils::calculate_stride(&bias_shape, &mut bias_stride);

                    let bias_data = bias
                        .data()
                        .and_then(|data| data.f32())
                        .ok_or(Error::MissingData)?;

                    trace!(
                        "x_data_len={:?},\
                        x_shape={x_shape:?},\
                        x_stride={x_stride:?},\
                        w_data_len={:?},\
                        w_shape={w_shape:?},\
                        pads={:?},\
                        strides={:?},\
                        dilations={:?},\
                        group={:?},\
                        y_data_len={:?},\
                        y_shape={y_shape:?},\
                        y_stride={y_stride:?}\
                        bias_shape={bias_shape:?},\
                        bias_stride={bias_stride:?}",
                        x_data.len(),
                        w_data.len(),
                        attrs.pads(),
                        attrs.strides(),
                        attrs.dilations(),
                        attrs.group(),
                        y_data.len(),
                    );

                    let bias = BiasInput {
                        data: bias_data,
                        shape: &bias_shape,
                        stride: &bias_stride,
                    };

                    rmlk_cuda::kernels::conv::compute::<f32>(
                        self.device,
                        (1.0, 0.0),
                        &x_data,
                        &x_shape,
                        &x_stride,
                        &w_data,
                        &w_shape,
                        attrs.pads(),
                        attrs.strides(),
                        attrs.dilations(),
                        attrs.group(),
                        Some(bias),
                        &mut y_data,
                        &y_shape,
                        &y_stride,
                    )
                    .map_err(|_| Error::ComputationFailed)?;
                }
            }

            let output = ctx.get_output_mut(0)?;
            output.init(CudaData::F32(y_data));
            output._reshape(y_shape.iter().map(|d| *d as usize).collect());
            output.set_dtype(DataType::Float);
        } else {
            return Err(Error::UnsupportedDataType);
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::conv::ConvKernel;
    use crate::test_utils;
    use crate::test_utils::{TestConvAttributes, TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};
    use std::collections::HashMap;

    #[test]
    fn test_conv_f32_2d_bias() {
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

        // Bias.
        let node_c = TestNode {
            shape: vec![1, 1, 1, 1],
            dtype,
            data: Some(CudaData::F32(device.htod_copy(vec![1.0; 1]).unwrap())),
        };

        let node_output = TestNode {
            shape,
            dtype,
            data: None,
        };

        let attributes = test_utils::create_conv_attributes(TestConvAttributes {
            dilations: Some(Box::new([1, 1])),
            group: Some(1),
            kernel_shape: None,
            pads: Some(Box::new([1, 1, 1, 1])),
            strides: Some(Box::new([1, 1])),
        });

        let params = TestParams {
            inputs: vec![node_a, node_b, node_c],
            outputs: vec![node_output],
            attributes,
            op: Op::Conv,
        };

        let mut state = test_utils::build_graph_and_state(params);
        let map = HashMap::from([(1, 1), (2, 2), (3, 3), (4, 4)]);
        let mut context = Context::new(&mut state, &map, 3).unwrap();

        let cuda_kernel = ConvKernel::new(device.clone());
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
                13.0, 22.0, 28.0, 34.0, 25.0, 34.0, 55.0, 64.0, 73.0, 52.0, 64.0, 100.0, 109.0,
                118.0, 82.0, 94.0, 145.0, 154.0, 163.0, 112.0, 73.0, 112.0, 118.0, 124.0, 85.0,
            ]
        )
    }

    #[test]
    fn test_conv_f32_2d() {
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

        let attributes = test_utils::create_conv_attributes(TestConvAttributes {
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

        let mut state = test_utils::build_graph_and_state(params);
        let map = HashMap::from([(1, 1), (2, 2), (3, 3)]);
        let mut context = Context::new(&mut state, &map, 2).unwrap();

        let cuda_kernel = ConvKernel::new(device.clone());
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
