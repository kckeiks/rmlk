use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use crate::utils;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
use log::trace;
use rmlk_schema::{DataType, Op};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct MaxPoolBackend<T> {
    device: Arc<CudaDevice>,
    _marker: PhantomData<T>,
}

impl<T> MaxPoolBackend<T>
where
    T: MaxPoolKernel,
{
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            _marker: PhantomData,
        }
    }
}

impl<T> MaxPoolBackend<T>
where
    T: MaxPoolKernel,
{
    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;

        let attrs = MaxPoolAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
        )?;

        let mut y_shape = scratch_alloc.allocate_fill(x_shape.len(), 0)?;

        // Todo: Move this to utils.
        // Todo: if attributes were usize, we wouldn't need to do this allocation here.
        rmlk_cuda::kernels::max_pool::compute_output_shape(
            &x_shape,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &mut y_shape,
            false,
        )?;

        let mut y_stride = scratch_alloc.allocate_fill(x_shape.len(), 0)?;
        utils::calculate_stride(&y_shape, &mut y_stride);

        trace!(
            "x_shape={x_shape:?},\
            x_stride={:?},\
            kernel_shape={:?},\
            pads={:?},\
            strides={:?}\
            y_shape={y_shape:?}\
            y_stride={y_stride:?}",
            x.stride(),
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides()
        );

        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;

        let dev_data = if matches!(x.dtype(), DataType::Float) {
            let x_dev_data_ref = x.dev_data().ok_or(InternalError::MissingDeviceData)?;
            let x_dev_data =
                x_dev_data_ref
                    .f32()
                    .ok_or_else(|| InternalError::UnexpectedTensorDataType {
                        expected: DataType::Float,
                    })?;

            // Todo: move this to DeviceService trait.
            let mut y_dev_data = self
                .device
                .alloc_zeros(y_shape.iter().map(|n| *n as usize).product())
                .map_err(rmlk_cuda::Error::from)?;

            T::execute::<f32>(
                self.device,
                1.0,
                0.0,
                &x_dev_data,
                &x_shape,
                &x_stride,
                attrs.kernel_shape(),
                attrs.pads(),
                attrs.strides(),
                &mut y_dev_data,
                &y_shape,
                &y_stride,
            )?;

            CudaData::F32(y_dev_data)
        } else {
            return Err(InternalError::UnsupportedOpForDataType {
                op: Op::GlobalAveragePool,
                dtype: *x.dtype(),
            });
        };

        let dtype = *x.dtype();
        let mut y = ctx.get_output_mut(0)?;
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        y.reshape(shape)?;
        y.set_dev_data(dev_data);
        y.set_dtype(dtype);

        Ok(())
    }
}

pub trait MaxPoolKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl MaxPoolKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::max_pool::compute::<T>(
            device,
            (alpha, beta),
            x_data,
            x_shape,
            x_stride,
            kernel_shape,
            pads,
            strides,
            y_data,
            y_shape,
            y_stride,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl MaxPoolKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: T,
        _: T,
        _: &CudaSlice<T>,
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &[i32],
        _: &[i32],
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
    use crate::providers::cuda::kernel::max_pool::BackendHandler;
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

        let cuda_kernel = BackendHandler::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .dev_data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 3.0, 4.0])
    }
}
