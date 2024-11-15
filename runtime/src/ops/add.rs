use crate::core::device_service::DeviceService;
use crate::core::kernel::{KernelError, Result};
use crate::core::Context;
use rmlk_schema::DataType;

pub trait Add {
    type Service: DeviceService;
    fn compute(
        self,
        lhs: &<Self::Service as DeviceService>::Data,
        lhs_shape: &[usize],
        lhs_dtype: DataType,
        rhs: &<Self::Service as DeviceService>::Data,
        rhs_shape: &[usize],
        rhs_dtype: DataType,
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

        let lhs_data = lhs
            .data()
            .ok_or_else(|| KernelError::Other("expected data in the tensor".to_string()))?;

        let rhs_data = rhs
            .data()
            .ok_or_else(|| KernelError::Other("expected data in the tensor".to_string()))?;

        let output = self.kernel.compute(
            lhs_data,
            lhs.shape().as_slice(),
            *lhs.dtype(),
            rhs_data,
            rhs.shape().as_slice(),
            *rhs.dtype(),
        )?;

        let result_shape = lhs.shape().clone();
        let result = ctx.get_output_mut(0)?;
        result.init(output);
        result._reshape(result_shape);
        result.set_dtype(DataType::Float);

        Ok(())
    }
}
