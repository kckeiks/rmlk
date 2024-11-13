use crate::core::device_service::DeviceService;
use crate::core::kernel::{KernelError, Result};
use crate::core::Context;
use log::trace;
use rmlk_schema::DataType;

pub trait Activation {
    type Data;
    fn compute(&self, x_data: &Self::Data, x_shape: &[i32], x_stride: &[i32])
        -> Result<Self::Data>;
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

    pub fn compute<D: DeviceService>(self, ctx: &mut Context<D>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        trace!("x_shape={x_shape:?},x_stride={x_stride:?}");

        let x_data = x.data().ok_or_else(|| {
            KernelError::Other("expected tensor data to be of type `float32`".to_string())
        })?;

        let y_data = self.kernel.compute(x_data, &x_shape, &x_stride)?;

        let output_shape = x.shape().clone();
        let output = ctx.get_output_mut(0)?;
        output.init(y_data);
        output._reshape(output_shape);
        output.set_dtype(DataType::Float);

        Ok(())
    }
}
