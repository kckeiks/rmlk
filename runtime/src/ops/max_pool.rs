use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::device_service::DeviceService;
use crate::core::error::{InternalError, Result};
use crate::core::{Context, ScratchAllocator, Tensor};
use crate::utils;
use log::trace;

pub trait MaxPoolBackend {
    type Service: DeviceService;

    // Todo: change all params to use usize.
    fn compute(
        self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        kernel_shape: &[i32],
        pads: &[i32],
        strides: &[i32],
        y_shape: &[i32],
        y_stride: &[i32],
        scratch_alloc: &ScratchAllocator,
    ) -> Result<<Self::Service as DeviceService>::Data>;
}

pub struct MaxPoolOp<T> {
    kernel: T,
}

impl<T> MaxPoolOp<T>
where
    T: MaxPoolBackend,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let x_shape = scratch_alloc.allocate_and_convert_from_slice(&x.shape())?;

        let attrs = MaxPoolAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
        )?;

        let mut y_shape = scratch_alloc.allocate_fill(x_shape.len(), 0)?;

        // Todo: Move this to utils.
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

        let dev_data = self.kernel.compute(
            &x,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &y_shape,
            &y_stride,
            ctx.execution_state().scratch_alloc(),
        )?;

        let y = ctx.get_output(0)?;
        let shape = scratch_alloc.allocate_and_convert_from_slice(y_shape)?;
        y.reshape(shape)?;

        let dtype = *x.dtype();
        let y = ctx.get_output_mut(0)?;
        y.init(dev_data);
        y.set_dtype(dtype);

        Ok(())
    }
}
