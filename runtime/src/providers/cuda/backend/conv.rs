use crate::attributes::conv::ConvAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use rmlk_cuda::kernels::conv::BiasInput;
use rmlk_schema::{DataType, Op};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct ConvolutionBackend<T> {
    device: Arc<CudaDevice>,
    kernel: PhantomData<T>,
}

impl<T> ConvolutionBackend<T>
where
    T: ConvolutionKernel,
{
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            kernel: PhantomData,
        }
    }
}

impl<T> ConvolutionBackend<T>
where
    T: ConvolutionKernel,
{
    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let filter_dims = match x.shape().len() {
            4 => 2,
            5 => 3,
            _ => {
                unreachable!("we already checked the dimensions of x for the supported dimensions")
            }
        };

        let attrs = ConvAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
            filter_dims,
        )?;

        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let mut y_shape = scratch_alloc.allocate_fill(x.shape().len(), 0)?;

        let w = ctx.get_input(1)?;
        let w_shape = scratch_alloc.allocate_and_convert_from_slice(&w.shape())?;

        rmlk_cuda::kernels::conv::calculate_output_shape(
            &x_shape,
            &w_shape,
            attrs.pads(),
            attrs.strides(),
            attrs.dilations(),
            &mut y_shape,
        )?;

        let mut y_stride = scratch_alloc.allocate_fill(x_shape.len(), 0)?;
        utils::calculate_stride(&y_shape, &mut y_stride);

        let bias = ctx.get_input(2).ok();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;
        let w_shape = scratch_alloc.allocate_and_convert_from_slice(&w.shape())?;

        // Extract and prepare bias argument.
        // At this point, we still don't know the data type of bias.
        let bias = match bias.as_ref() {
            Some(bias_tensor) => {
                let bias_shape = scratch_alloc.allocate_fill(x_shape.len(), 1)?;
                // Todo: Urgent. We need to make this generic.
                bias_shape[1] = bias_tensor.shape()[0] as i32;

                let bias_stride = scratch_alloc.allocate_fill(x_shape.len(), 0)?;
                utils::calculate_stride(&bias_shape, bias_stride);

                let device_data =
                    bias_tensor
                        .data()
                        .ok_or_else(|| InternalError::UnexpectedTensorDataType {
                            expected: DataType::Float,
                        })?;

                Some(BiasArg {
                    data: device_data,
                    shape: bias_shape,
                    stride: bias_stride,
                })
            }
            None => None,
        };

        let dev_data = if matches!(x.dtype(), DataType::Float) {
            let x_data = x.data().and_then(|data| data.f32()).ok_or_else(|| {
                InternalError::UnexpectedTensorDataType {
                    expected: DataType::Float,
                }
            })?;
            let w_data = w.data().and_then(|data| data.f32()).ok_or_else(|| {
                InternalError::UnexpectedTensorDataType {
                    expected: DataType::Float,
                }
            })?;

            let mut y_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|d| *d as usize).product())
                .map_err(rmlk_cuda::Error::from)?;

            // Since we know the data type, we extract it.
            let float_bias = match bias {
                Some(bias) => {
                    let data =
                        bias.data
                            .f32()
                            .ok_or_else(|| InternalError::UnexpectedTensorDataType {
                                expected: DataType::Float,
                            })?;
                    Some(BiasInput {
                        data,
                        shape: bias.shape,
                        stride: bias.stride,
                    })
                }
                None => None,
            };

            T::execute::<f32>(
                self.device,
                1.0,
                0.0,
                &x_data,
                &x_shape,
                &x_stride,
                &w_data,
                &w_shape,
                attrs.pads(),
                attrs.strides(),
                attrs.dilations(),
                attrs.group(),
                float_bias,
                &mut y_data,
                &y_shape,
                &y_stride,
            )?;

            CudaData::F32(y_data)
        } else {
            return Err(InternalError::UnsupportedOpForDataType {
                op: Op::Conv,
                dtype: *x.dtype(),
            });
        };

        let dtype = *x.dtype();
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        let mut y = ctx.get_output_mut(0)?;
        y.reshape(shape)?;
        y.init(dev_data);
        y.set_dtype(dtype);

        Ok(())
    }
}

struct BiasArg<'a, T> {
    data: T,
    shape: &'a [i32],
    stride: &'a [i32],
}

pub trait ConvolutionKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        w_data: &CudaSlice<T>,
        w_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        dilations: &[i32],
        group: i32,
        bias: Option<BiasInput<T>>,
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl ConvolutionKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        w_data: &CudaSlice<T>,
        w_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        dilations: &[i32],
        group: i32,
        bias: Option<BiasInput<T>>,
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::conv::compute::<T>(
            device,
            (alpha, beta),
            x_data,
            x_shape,
            x_stride,
            w_data,
            w_shape,
            pads,
            strides,
            dilations,
            group,
            bias,
            y_data,
            y_shape,
            y_stride,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl ConvolutionKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: T,
        _: T,
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: i32,
        _: Option<BiasInput<T>>,
        _: &mut CudaSlice<T>,
        _: &[i32],
        _: &[i32],
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::conv::BackendHandler;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestConvAttributes, TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

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

        let attributes = test_utils::create_conv_attributes(TestConvAttributes {
            dilations: Some(Box::new([1, 1])),
            group: Some(1),
            kernel_shape: None,
            pads: Some(Box::new([1, 1, 1, 1])),
            strides: Some(Box::new([1, 1])),
        });

        let params = TestParams {
            inputs: vec![node_a, node_b, node_c],
            attributes,
            op: Op::Conv,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 4).unwrap();

        let cuda_kernel = BackendHandler::new(device.clone());
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

        let attributes = test_utils::create_conv_attributes(TestConvAttributes {
            dilations: Some(Box::new([1, 1])),
            group: Some(1),
            kernel_shape: None,
            pads: Some(Box::new([1, 1, 1, 1])),
            strides: Some(Box::new([1, 1])),
        });

        let params = TestParams {
            inputs: vec![node_a, node_b],
            attributes,
            op: Op::Conv,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 3).unwrap();

        let cuda_kernel = BackendHandler::new(device.clone());
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
