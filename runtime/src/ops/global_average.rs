use crate::core::device_service::{DeviceService, DeviceServiceError};
use crate::core::kernel::{KernelError, Result};
use crate::core::{Context, Tensor};
use crate::utils;
use rmlk_schema::DataType;

pub trait GlobalAverage {
    type Service: DeviceService;
    fn compute(
        self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        y_shape: &[usize],
        y_stride: &[usize],
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

        let mut y_shape = vec![0; x.shape().len()];
        // Todo: move this to utils.
        rmlk_cuda::kernels::global_average_pool::compute_output_shape(
            x.shape().as_slice(),
            y_shape.as_mut_slice(),
        )?;

        let mut y_stride = vec![0; y_shape.len()];
        utils::calculate_stride(&y_shape, &mut y_stride);

        let y_data = self.kernel.compute(&x, &y_shape, &y_stride)?;

        let dtype = *x.dtype();
        let output = ctx.get_output_mut(0)?;
        output.init(y_data);
        output._reshape(y_shape.iter().map(|d| *d as usize).collect());
        output.set_dtype(dtype);

        Ok(())
    }
}
