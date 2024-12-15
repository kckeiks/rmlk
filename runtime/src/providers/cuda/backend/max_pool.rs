use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct MaxPoolBackend {
    device: Arc<CudaDevice>,
}

impl MaxPoolBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

impl MaxPoolBackend {
    fn comput_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;

        let attrs = MaxPoolAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
            ctx.execution_state().scratch_alloc(),
        )?;

        let mut y_shape = scratch_alloc.allocate_fill(x_shape.len(), 0)?;

        // Todo: Refactor function so we dont have to allocate a scratch buffer.
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

        let y = ctx.get_output(0)?;
        let y_index = y.dst_id();
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        ctx.execution_state_mut()
            .copy_shape_from_slice(shape, y_index)?;

        Ok(())
    }

    fn compute_max_pool<D, T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: MaxPoolKernel,
    {
        self.comput_output_shape(ctx)?;

        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;

        let attrs = MaxPoolAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
            ctx.execution_state().scratch_alloc(),
        )?;

        debug!(
            "[x][max_pool][shape={:?}][stride=[{:?}]",
            x.shape(),
            x.stride()
        );
        debug!(
            "[y][max_pool][shape={:?}][stride=[{:?}]",
            y.shape(),
            y.stride()
        );
        debug!(
            "[max_pool][pads={:?}][strides=[{:?}][kernel_shape={:?}]",
            attrs.pads(),
            attrs.strides(),
            attrs.kernel_shape()
        );

        let scratch_alloc = ctx.execution_state().scratch_alloc();

        let x_shape = scratch_alloc.allocate_and_convert_from_slice(x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(x.stride())?;
        let y_shape = scratch_alloc.allocate_and_convert_from_slice(y.shape())?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(y.stride())?;

        let x_dev_data_ref = x.try_dev_data_ptr()?;
        let x_dev_data = x_dev_data_ref.data();

        let elem_count = y_shape.iter().map(|n| *n as usize).product();

        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let mut y = ctx.get_output(0)?;
            let y_dev_data_ref = y.dev_data_ptr_mut();
            let need_to_alloc_dev_data = y_dev_data_ref.is_none()
                || y_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != elem_count)
                    .unwrap_or(true);

            // We need to remove this immutable reference so we can mutate `y`.
            drop(y_dev_data_ref);

            if need_to_alloc_dev_data {
                let y_dev_data = self
                    .device
                    .alloc_zeros::<D>(elem_count)
                    .map_err(rmlk_cuda::Error::from)?;
                y.set_dev_data(CudaData::new(y_dev_data));
            };
        }

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        T::execute::<D>(
            self.device.clone(),
            D::one(),
            D::zero(),
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

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: MaxPoolKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_max_pool::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::MaxPool,
                dtype,
            }),
        }
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
            .dev_data_ptr()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 3.0, 4.0])
    }
}
