use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::device_service::{DeviceService, DeviceServiceError};
use crate::core::kernel::{KernelError, Result};
use crate::core::Context;
use crate::utils;
use log::trace;
use rmlk_schema::DataType;
use std::sync::Arc;

pub trait MaxPool {
    type Data;
    fn compute(
        self,
        x_data: &Self::Data,
        x_shape: &[i32],
        x_stride: &[i32],
        kernel_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<Self::Data>;
}

pub struct MaxPoolOp<T> {
    kernel: T,
}

impl<T> MaxPoolOp<T>
where
    T: MaxPool,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute<D: DeviceService>(self, ctx: &mut Context<D>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let attrs =
            MaxPoolAttributes::new(ctx.get_attributes().ok_or(KernelError::MissingAttributes)?)?;

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();

        // Todo: Move this to utils.
        rmlk_cuda::kernels::max_pool::compute_output_shape(
            &x_shape,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &mut y_shape,
            false,
        )?;

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

        let x_data = x.data().ok_or_else(|| {
            KernelError::Other("expected tensor data to be of type `float32`".to_string())
        })?;

        // Todo: move this to DeviceService trait.
        let y_data = self.kernel.compute(
            &x_data,
            &x_shape,
            &x_stride,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &y_shape,
            &y_stride,
        )?;

        let output_dtype = *x.dtype();
        let output = ctx.get_output_mut(0)?;
        output.init(y_data);
        output._reshape(y_shape.iter().map(|d| *d as usize).collect());
        output.set_dtype(output_dtype);

        Ok(())
    }
}
