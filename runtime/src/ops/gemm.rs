use crate::attributes::gemm::GemmAttributes;
use crate::core::device_service::DeviceService;
use crate::core::kernel::{KernelError, Result};
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
        let lhs = ctx.get_input(0)?;
        let rhs = ctx.get_input(1)?;

        let attrs =
            GemmAttributes::new(ctx.get_attributes().ok_or(KernelError::MissingAttributes)?)?;

        let (out_data, output_shape) = self.kernel.compute(
            lhs,
            rhs,
            attrs.trans_a(),
            attrs.trans_b(),
            attrs.alpha(),
            attrs.beta(),
            ctx.execution_state().scratch_alloc(),
        )?;

        let output_dtype = *lhs.dtype();
        let output = ctx.get_output_mut(0)?;
        output.init(out_data);
        output._reshape(output_shape.to_vec().into_boxed_slice());
        output.set_dtype(output_dtype);

        Ok(())
    }
}
