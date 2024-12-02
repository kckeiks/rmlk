use crate::core::device_service::DeviceService;
use crate::core::error::Result;
use crate::core::{Context, ScratchAllocator, Tensor};
use crate::utils;

pub trait GlobalAverageBackend {
    type Service: DeviceService;
    fn compute(
        self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        y_shape: &[usize],
        y_stride: &[usize],
        scratch_alloc: &ScratchAllocator,
    ) -> Result<<Self::Service as DeviceService>::Data>;
}

pub struct GlobalAverageOp<T> {
    kernel: T,
}

impl<T> GlobalAverageOp<T>
where
    T: GlobalAverageBackend,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let x = ctx.get_input(0)?;

        let y_shape = ctx
            .execution_state()
            .scratch_alloc()
            .allocate_fill(x.shape().len(), 0)?;
        // Todo: move this to utils.
        rmlk_cuda::kernels::global_average_pool::compute_output_shape(&x.shape(), y_shape)?;

        let y_stride = ctx
            .execution_state()
            .scratch_alloc()
            .allocate_fill(y_shape.len(), 0)?;
        utils::calculate_stride(&y_shape, y_stride);

        let dev_data = self.kernel.compute(
            &x,
            &y_shape,
            &y_stride,
            ctx.execution_state().scratch_alloc(),
        )?;

        let y = ctx.get_output(0)?;
        y.reshape(y_shape)?;

        let dtype = *x.dtype();
        let y = ctx.get_output_mut(0)?;
        y.init(dev_data);
        y.set_dtype(dtype);

        Ok(())
    }
}
