use crate::attributes::pooling::MaxPoolAttributes;
use crate::core::device_service::DeviceService;
use crate::core::error::{InternalError, Result};
use crate::core::{Context, ScratchAllocator, Tensor};
use crate::utils;
use log::trace;

pub trait MaxPool {
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
    T: MaxPool,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let attrs = MaxPoolAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
        )?;

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

        // Todo: move this to DeviceService trait.
        let y_data = self.kernel.compute(
            &x,
            attrs.kernel_shape(),
            attrs.pads(),
            attrs.strides(),
            &y_shape,
            &y_stride,
            ctx.execution_state().scratch_alloc(),
        )?;

        let output_dtype = *x.dtype();
        let output = ctx.get_output_mut(0)?;
        output.init(y_data);
        output._reshape(y_shape.iter().map(|d| *d as usize).collect());
        output.set_dtype(output_dtype);

        Ok(())
    }
}
