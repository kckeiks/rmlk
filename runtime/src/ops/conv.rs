use crate::attributes::conv::ConvAttributes;
use crate::core::device_service::DeviceService;
use crate::core::error::InternalError;
use crate::core::kernel::Result;
use crate::core::{Context, ScratchAllocator, Tensor};
use crate::utils;
use rmlk_schema::DataType;

pub struct BiasInput<'a, T> {
    pub data: &'a T,
    pub shape: &'a [i32],
    pub stride: &'a [i32],
}

pub trait Convolution {
    type Service: DeviceService;
    fn compute(
        self,
        x: &Tensor<<Self::Service as DeviceService>::Data>,
        w: &Tensor<<Self::Service as DeviceService>::Data>,
        pads: &[i32],
        strides: &[i32],
        dilations: &[i32],
        group: i32,
        bias: Option<BiasInput<<Self::Service as DeviceService>::Data>>,
        y_shape: &[i32],
        y_stride: &[i32],
        scratch_alloc: &ScratchAllocator,
    ) -> Result<<Self::Service as DeviceService>::Data>;
}

pub struct ConvolutionOp<T> {
    kernel: T,
}

impl<T> ConvolutionOp<T>
where
    T: Convolution,
{
    pub fn new(kernel: T) -> Self {
        Self { kernel }
    }

    pub fn compute(self, ctx: &mut Context<T::Service>) -> Result<()> {
        let x = ctx.get_input(0)?;
        // Todo: reuse buffers.
        let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let w = ctx.get_input(1)?;

        let filter_dims = match x.shape().len() {
            4 => 2,
            5 => 3,
            _ => {
                unreachable!("we already checked the dimensions of x for the supported dimensions")
            }
        };

        let attrs = ConvAttributes::new(
            ctx.get_attributes()
                .ok_or(InternalError::MissingAttributes)?,
            filter_dims,
        )?;

        // Todo: reuse buffers.
        let w_shape = w.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

        let mut y_shape = vec![0; x_shape.len()].into_boxed_slice();
        rmlk_cuda::kernels::conv::calculate_output_shape(
            &x_shape,
            &w_shape,
            attrs.pads(),
            attrs.strides(),
            attrs.dilations(),
            &mut y_shape,
        )?;

        let mut y_stride = vec![0; x_shape.len()].into_boxed_slice();
        utils::calculate_stride(&y_shape, &mut y_stride);

        let y_data = match ctx.get_input(2).ok() {
            None => self.kernel.compute(
                &x,
                &w,
                attrs.pads(),
                attrs.strides(),
                attrs.dilations(),
                attrs.group(),
                None,
                &y_shape,
                &y_stride,
                ctx.execution_state().scratch_alloc(),
            )?,
            Some(bias) => {
                let mut bias_shape = vec![1i32; x_shape.len()];
                // Todo: Urgent. We need to make this generic.
                bias_shape[1] = bias.shape()[0] as i32;

                let mut bias_stride = vec![0i32; x_shape.len()];
                utils::calculate_stride(&bias_shape, &mut bias_stride);

                let bias_data =
                    bias.data()
                        .ok_or_else(|| InternalError::UnexpectedTensorDataType {
                            expected: DataType::Float,
                        })?;

                let bias = BiasInput {
                    data: bias_data,
                    shape: &bias_shape,
                    stride: &bias_stride,
                };

                self.kernel.compute(
                    &x,
                    &w,
                    attrs.pads(),
                    attrs.strides(),
                    attrs.dilations(),
                    attrs.group(),
                    Some(bias),
                    &y_shape,
                    &y_stride,
                    ctx.execution_state().scratch_alloc(),
                )?
            }
        };

        let output_dtype = *x.dtype();

        let output = ctx.get_output_mut(0)?;
        output.init(y_data);
        output._reshape(y_shape.iter().map(|d| *d as usize).collect());
        output.set_dtype(output_dtype);

        Ok(())
    }
}
