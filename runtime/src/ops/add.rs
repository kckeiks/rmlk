use crate::core::device_service::DeviceService;
use crate::core::error::Result;
use crate::core::{Context, ScratchAllocator, Tensor};

pub trait Add {
    type Service: DeviceService;
    fn compute(
        self,
        lhs: &Tensor<<Self::Service as DeviceService>::Data>,
        rhs: &Tensor<<Self::Service as DeviceService>::Data>,
        scratch_alloc: &ScratchAllocator,
    ) -> Result<<Self::Service as DeviceService>::Data>;
}

pub struct AddOp<T> {
    kernel: T,
}

impl<T> AddOp<T>
where
    T: Add,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        debug_assert!(a.shape().as_ref() == b.shape().as_ref());

        let dev_data = self
            .kernel
            .compute(a, b, ctx.execution_state().scratch_alloc())?;

        let c = ctx.get_output(0)?;
        c.reshape(&a.shape())?;

        let dtype = *a.dtype();
        let c = ctx.get_output_mut(0)?;
        c.init(dev_data);
        c.set_dtype(dtype);

        Ok(())
    }
}
