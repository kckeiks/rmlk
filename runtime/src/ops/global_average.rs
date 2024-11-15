use crate::core::device_service::{DeviceService, DeviceServiceError};
use crate::core::kernel::{KernelError, Result};
use crate::core::Context;
use crate::utils;
use rmlk_schema::DataType;

pub trait GlobalAverage {
    type Service: DeviceService;
    fn compute(
        self,
        x_data: &<Self::Service as DeviceService>::Data,
        x_shape: &[i32],
        x_stride: &[i32],
        y_shape: &[i32],
        y_stride: &[i32],
    ) -> Result<<Self::Service as DeviceService>::Data>;
}

pub struct GlobalAverageOp<T> {
    kernel: T,
}

impl<T> GlobalAverageOp<T>
where
    T: GlobalAverage,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let x = ctx.get_input(0)?;
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
        let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let mut y_shape = vec![0; x_shape.len()];
        // Todo: move this to utils.
        rmlk_cuda::kernels::global_average_pool::compute_output_shape(
            &x_shape,
            y_shape.as_mut_slice(),
        )?;

        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);

        let x_data = x.data().ok_or_else(|| {
            KernelError::Other("expected tensor data to be of type `float32`".to_string())
        })?;

        let y_data = self
            .kernel
            .compute(&x_data, &x_shape, &x_stride, &y_shape, &y_stride)?;

        let dtype = x.dtype();
        let output = ctx.get_output_mut(0)?;
        output.init(y_data);
        output._reshape(y_shape.iter().map(|d| *d as usize).collect());
        output.set_dtype(*dtype);

        Ok(())
    }
}
