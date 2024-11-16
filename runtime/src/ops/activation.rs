use crate::core::device_service::DeviceService;
use crate::core::kernel::Result;
use crate::core::{Context, Tensor};
use log::trace;
use rmlk_schema::DataType;

pub trait Activation {
    type Service: DeviceService;
    fn compute(
        &self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
    ) -> Result<<Self::Service as DeviceService>::Data>;
}

pub struct ActivationOp<T> {
    kernel: T,
}

impl<T> ActivationOp<T>
where
    T: Activation,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let x = ctx.get_input(0)?;

        trace!("x_shape={:?},x_stride={:?}", x.shape(), x.stride());

        let y_data = self.kernel.compute(x)?;

        let output_shape = x.shape().clone();
        let output = ctx.get_output_mut(0)?;
        output.init(y_data);
        output._reshape(output_shape);
        output.set_dtype(DataType::Float);

        Ok(())
    }
}
