use crate::core::device_service::DeviceService;
use crate::core::error::Result;
use crate::core::{Context, ScratchAllocator, Tensor};
use log::trace;

pub trait ActivationBackend {
    type Service: DeviceService;
    fn compute(
        &self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        scratch_alloc: &ScratchAllocator,
    ) -> Result<<Self::Service as DeviceService>::Data>;
}

pub struct ActivationOp<T> {
    kernel: T,
}

impl<T> ActivationOp<T>
where
    T: ActivationBackend,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let x = ctx.get_input(0)?;

        trace!("x_shape={:?},x_stride={:?}", x.shape(), x.stride());

        let dev_data = self
            .kernel
            .compute(x, ctx.execution_state().scratch_alloc())?;

        let y = ctx.get_output(0)?;
        y.reshape(&x.shape())?;

        let dtype = *x.dtype();
        let y = ctx.get_output_mut(0)?;
        y.init(dev_data);
        y.set_dtype(dtype);

        Ok(())
    }
}
