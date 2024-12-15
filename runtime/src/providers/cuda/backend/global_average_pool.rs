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

pub struct GlobalAverageBackend {
    device: Arc<CudaDevice>,
}

impl GlobalAverageBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

impl GlobalAverageBackend {
    fn comput_output_shape(&self, ctx: &mut Context<Cuda>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc().clone();

        let y_shape_original = scratch_alloc.allocate_fill(x.shape().len(), 0)?;
        // Todo: move this to utils.
        rmlk_cuda::kernels::global_average_pool::compute_output_shape(
            &x.shape(),
            y_shape_original,
        )?;

        let y = ctx.get_output(0)?;
        let y_index = y.dst_id();
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape_original)?;
        ctx.execution_state_mut()
            .copy_shape_from_slice(shape, y_index)?;

        Ok(())
    }

    fn compute_global_average_pool<D, T>(&self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: GlobalAveragePoolKernel,
    {
        self.comput_output_shape(ctx)?;

        let x = ctx.get_input(0)?;
        let y = ctx.get_output(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc();

        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;
        let x_stride = scratch_alloc.allocate_and_convert_from_slice(&x.stride())?;

        let y_shape = scratch_alloc.allocate_and_convert_from_slice(y.shape())?;
        let y_stride = scratch_alloc.allocate_and_convert_from_slice(y.stride())?;

        let pads = scratch_alloc.allocate_fill(x_shape[2..].len(), 0)?;
        let strides = scratch_alloc.allocate_fill(x_shape[2..].len(), 1)?;
        let kernel_shape = &x_shape[2..];

        debug!(
            "[x][global_avg_pool][shape={:?}][stride=[{:?}]",
            x.shape(),
            x.stride()
        );
        debug!(
            "[y][global_avg_pool][shape={:?}][stride=[{:?}]",
            y.shape(),
            y.stride()
        );
        debug!(
            "[global_avg_pool][pads={:?}][strides=[{:?}][kernel_shape={:?}]",
            pads, strides, kernel_shape
        );

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
            pads,
            strides,
            &x_dev_data,
            &x_shape,
            &x_stride,
            kernel_shape,
            &mut y_dev_data,
            &y_shape,
            &y_stride,
        )?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: GlobalAveragePoolKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_global_average_pool::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::GlobalAveragePool,
                dtype,
            }),
        }
    }
}

pub trait GlobalAveragePoolKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        pads: &[i32],
        strides: &[i32],
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl GlobalAveragePoolKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        alpha: T,
        beta: T,
        pads: &[i32],
        strides: &[i32],
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        y_data: &mut CudaSlice<T>,
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::global_average_pool::compute::<T>(
            device,
            (alpha, beta),
            pads,
            strides,
            x_data,
            x_shape,
            x_stride,
            kernel_shape,
            y_data,
            y_shape,
            y_stride,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl GlobalAveragePoolKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: T,
        _: T,
        _: &[i32],
        _: &[i32],
        _: &CudaSlice<T>,
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
    use crate::providers::cuda::kernel::global_average_pool::BackendHandler;
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

        assert_eq!(result, vec![5.0])
    }
}
