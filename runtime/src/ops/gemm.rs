use crate::attributes::gemm::GemmAttributes;
use crate::core::device_service::DeviceService;
use crate::core::kernel::{KernelError, Result};
use crate::core::Context;

pub trait Gemm {
    type Data;
    // Todo: refactor so you can remove the shape output.
    fn compute(
        self,
        lhs: &Self::Data,
        lhs_shape: &[usize],
        lhs_stride: &[usize],
        rhs: &Self::Data,
        rhs_shape: &[usize],
        rhs_stride: &[usize],
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<(Self::Data, [usize; 3])>;
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

    pub fn compute<D: DeviceService>(self, ctx: &mut Context<D>) -> Result<()> {
        let lhs = ctx.get_input(0)?;
        let rhs = ctx.get_input(1)?;

        let attrs =
            GemmAttributes::new(ctx.get_attributes().ok_or(KernelError::MissingAttributes)?)?;

        let lhs_data = lhs.data().ok_or_else(|| {
            KernelError::Other("expected lhs tensor data to be of type `float32`".to_string())
        })?;
        let rhs_data = rhs.data().ok_or_else(|| {
            KernelError::Other("expected rhs tensor data to be of type `float32`".to_string())
        })?;

        let (out_data, output_shape) = self.kernel.compute(
            lhs_data,
            lhs.shape(),
            lhs.stride(),
            rhs_data,
            rhs.shape(),
            rhs.stride(),
            attrs.trans_a(),
            attrs.trans_b(),
            attrs.alpha(),
            attrs.beta(),
        )?;

        let output_dtype = lhs.dtype();
        let output = ctx.get_output_mut(0)?;
        output.init(out_data);
        output._reshape(output_shape.to_vec());
        output.set_dtype(*output_dtype);

        Ok(())
    }
}
