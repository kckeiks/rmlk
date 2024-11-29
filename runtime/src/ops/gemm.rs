use crate::attributes::gemm::GemmAttributes;
use crate::core::device_service::DeviceService;
use crate::core::error::{InternalError, Result};
use crate::core::{Context, ScratchAllocator, Tensor};

pub trait Gemm {
    type Service: DeviceService;
    // Todo: refactor so you can remove the shape output.
    fn compute(
        self,
        lhs: &Tensor<<Self::Service as DeviceService>::Data>,
        rhs: &Tensor<<Self::Service as DeviceService>::Data>,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
        scratch_alloc: &ScratchAllocator,
    ) -> Result<(<Self::Service as DeviceService>::Data, [usize; 3])>;
}

pub struct GemmOp<T> {
    kernel: T,
}

impl<T> GemmOp<T>
where
    T: Gemm,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        let attrs = GemmAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
        )?;

        let (dev_data, y_shape) = self.kernel.compute(
            a,
            b,
            attrs.trans_a(),
            attrs.trans_b(),
            attrs.alpha(),
            attrs.beta(),
            ctx.execution_state().scratch_alloc(),
        )?;

        let y = ctx.get_output(0)?;
        y.reshape(y_shape.as_slice())?;

        let y_dtype = *a.dtype();
        let y = ctx.get_output_mut(0)?;
        y.init(dev_data);
        y.set_dtype(y_dtype);

        Ok(())
    }
}
