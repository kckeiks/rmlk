use crate::core::device_service::DeviceService;
use crate::core::kernel::{KernelError, Result};
use crate::core::{Context, Tensor};
use rmlk_schema::DataType;

pub trait Add {
    type Service: DeviceService;
    fn compute(
        self,
        lhs: &Tensor<<Self::Service as DeviceService>::Data>,
        rhs: &Tensor<<Self::Service as DeviceService>::Data>,
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
        let lhs = ctx.get_input(0)?;
        let rhs = ctx.get_input(1)?;

        debug_assert!(lhs.shape() == rhs.shape());

        let output = self.kernel.compute(lhs, rhs)?;

        let result_shape = lhs.shape().clone();
        let result = ctx.get_output_mut(0)?;
        result.init(output);
        result._reshape(result_shape);
        result.set_dtype(DataType::Float);

        Ok(())
    }
}
