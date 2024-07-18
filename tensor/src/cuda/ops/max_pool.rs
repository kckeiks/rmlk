use crate::cuda::data::CudaData;
use crate::kernel::Allocator;
use crate::{Context, Error, Result};
use cudarc::cudnn::{Cudnn, PoolingForward};
use cudarc::driver::CudaDevice;
use log::debug;
use num_traits::cast::FromPrimitive;
use rmlk_ir::{Attribute, DataType};
use std::collections::HashMap;
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    // Input data tensor.
    let x = ctx.get_input(0)?;

    if x.shape().len() != 4 && x.shape().len() != 5 {
        return Err(Error::UnsupportedShape);
    }

    // Todo: let's fix the casting here.
    let x_shape = x.shape().iter().map(|d| *d as i32).collect::<Box<[i32]>>();
    let x_stride = x.stride().iter().map(|d| *d as i32).collect::<Box<[i32]>>();

    let kernel_dims = match x.shape().len() {
        4 => 2,
        5 => 3,
        _ => unreachable!("we already checked the dimensions of x for the supported dimensions"),
    };

    let attrs = MaxPoolAttributes::new(
        ctx.get_attributes().ok_or(Error::MissingAttributes)?,
        kernel_dims,
    )?;

    let out_shape = attrs.calculate_output_shape(x_shape.as_ref())?;

    let out = ctx.get_output(0)?;
    if out.shape().len() != out_shape.len() {
        return Err(Error::OutputShapeMismatch);
    }

    // Todo: if there is no shape, simply set it.
    // Although, we probably need some other type of way
    // to flag tensors when they could have any shape.
    if out
        .shape()
        .iter()
        .zip(out_shape.iter())
        .map(|(a, b)| (*a, *b as usize))
        .find(|(a, b)| a != b)
        .is_some()
    {
        return Err(Error::OutputShapeMismatch);
    }

    let out_stride = out
        .stride()
        .iter()
        .map(|v| *v as i32)
        .collect::<Box<[i32]>>();
    let out_slice_size = out_shape.iter().product::<i32>();

    debug!(
        "x_shape={x_shape:?}, \
        pads={:?}, \
        strides={:?}",
        attrs.pads(),
        attrs.strides()
    );

    match x.dtype() {
        DataType::Float => {
            let x_desc = cudnn
                .create_nd_tensor::<f32>(&x_shape, &x_stride)
                .map_err(|_| Error::CudnnInternal)?;

            let pooling = cudnn
                .create_poolingnd::<f32>(
                    attrs.kernel_shape(),
                    // Todo: Let's preprocess pads.
                    attrs.pads(),
                    attrs.strides(),
                    cudarc::cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
                    cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                )
                .map_err(|_| Error::CudnnInternal)
                .unwrap();

            let out_desc = cudnn
                .create_nd_tensor(out_shape.as_ref(), out_stride.as_ref())
                .map_err(|_| Error::CudnnInternal)?;
            let mut out_slice = device
                .alloc_zeros::<f32>(usize::try_from(out_slice_size).unwrap())
                .map_err(|_| Error::CudnnInternal)?;

            let forward_f = PoolingForward {
                pooling: &pooling,
                x: &x_desc,
                y: &out_desc,
            };

            forward_f
                .launch(
                    (1.0, 0.0),
                    x.data().ok_or(Error::MissingTensor)?.f32()?,
                    &mut out_slice,
                )
                .map_err(|_| Error::CudnnInternal)?;

            let out_tensor = ctx.get_output_mut(0)?;
            out_tensor.init(CudaData::F32(out_slice));
        }
        _ => return Err(Error::UnsupportedDataType),
    }

    // Todo: Finish.
    Ok(())
}

pub struct MaxPoolAttributes {
    ceil_mode: bool,
    _dilations: Box<[i32]>,
    kernel_shape: Box<[i32]>,
    pads: Box<[i32]>,
    _row_major_order: bool,
    strides: Box<[i32]>,
    kernel_dims: usize,
    alloc: Allocator,
}

impl MaxPoolAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>, kernel_dims: usize) -> Result<Self> {
        let kernel_shape = match attrs.get("kernel_shape") {
            Some(attr) => Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    // Todo: Let's figure out to avoid these allocations.
                    .to_vec()
                    .into_boxed_slice(),
            ),
            None => return Err(Error::MissingAttributes),
        };

        let alloc = Allocator;
        let mut ceil_mode = None;
        let mut pads = None;
        let mut strides = None;
        let mut row_major_order = None;

        if let Some(attr) = attrs.get("ceil_mode") {
            ceil_mode = match attr.int() {
                Some(0) => Some(false),
                Some(_) => return Err(Error::UnsupportedAttribute),
                None => None,
            }
        }

        if attrs.get("dilations").is_some() {
            return Err(Error::UnsupportedAttribute);
        }

        if let Some(attr) = attrs.get("pads") {
            pads = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        if let Some(attr) = attrs.get("strides") {
            strides = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        if let Some(attr) = attrs.get("row_major_order") {
            row_major_order = match attr.int() {
                Some(0) => Some(true),
                _ => None,
            }
        }

        Ok(Self {
            _dilations: alloc.alloc_with_value::<i32>(1, kernel_dims),
            ceil_mode: ceil_mode.unwrap_or(false),
            kernel_shape: kernel_shape.ok_or(Error::MissingAttributes)?,
            pads: pads.unwrap_or_else(|| alloc.alloc_with_value::<i32>(0, kernel_dims)),
            _row_major_order: row_major_order.unwrap_or(false),
            strides: strides.unwrap_or_else(|| alloc.alloc_with_value::<i32>(1, kernel_dims)),
            kernel_dims,
            alloc,
        })
    }

    pub fn pads(&self) -> &[i32] {
        self.pads.as_ref()
    }

    pub fn strides(&self) -> &[i32] {
        self.strides.as_ref()
    }

    pub fn kernel_shape(&self) -> &[i32] {
        self.kernel_shape.as_ref()
    }

    pub fn calculate_output_shape(&self, x_shape: &[i32]) -> Result<Box<[i32]>> {
        // For reference, https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-ops-library.html#cudnngetpoolingndforwardoutputdim.
        if self.kernel_dims == 2 {
            let height = (f64::from(x_shape[2] + 2 * self.pads[0] - self.kernel_shape[0])
                / self.strides[0] as f64)
                + 1.0;
            let height = match self.ceil_mode {
                true => height.ceil(),
                false => height.floor(),
            };

            let width = (f64::from(x_shape[3] + 2 * self.pads[1] - self.kernel_shape[1])
                / self.strides[1] as f64)
                + 1.0;
            let width = match self.ceil_mode {
                true => width.ceil(),
                false => width.floor(),
            };

            let mut buf = self.alloc.alloc_with_value(0, 4);
            buf[0] = x_shape[0];
            buf[1] = x_shape[1];
            buf[2] = i32::from_f64(height).ok_or(Error::ComputationError)?;
            buf[3] = i32::from_f64(width).ok_or(Error::ComputationError)?;

            Ok(buf)
        } else if self.kernel_dims == 3 {
            let depth = (f64::from(x_shape[1] + 2 * self.pads[0] - self.kernel_shape[0])
                / self.strides[0] as f64)
                + 1.0;
            let depth = match self.ceil_mode {
                true => depth.ceil(),
                false => depth.floor(),
            };

            let height = (f64::from(x_shape[2] + 2 * self.pads[1] - self.kernel_shape[1])
                / self.strides[1] as f64)
                + 1.0;
            let height = match self.ceil_mode {
                true => height.ceil(),
                false => height.floor(),
            };

            let width = (f64::from(x_shape[3] + 2 * self.pads[2] - self.kernel_shape[2])
                / self.strides[2] as f64)
                + 1.0;
            let width = match self.ceil_mode {
                true => width.ceil(),
                false => width.floor(),
            };

            let mut buf = self.alloc.alloc_with_value(0, 5);
            buf[0] = x_shape[0];
            buf[1] = x_shape[1];
            buf[2] = i32::from_f64(depth).ok_or(Error::ComputationError)?;
            buf[3] = i32::from_f64(height).ok_or(Error::ComputationError)?;
            buf[4] = i32::from_f64(width).ok_or(Error::ComputationError)?;

            Ok(buf)
        } else {
            unreachable!("Constructor method validates that only 2d and 3d are supported");
        }
    }

    pub fn _calculate_output_shape_pytorch(&self, x_shape: &[i32]) -> Result<Box<[i32]>> {
        if self.kernel_dims == 2 {
            // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d.
            let height = (f64::from(
                x_shape[2] + 2 * self.pads[0] - self._dilations[0] * (self.kernel_shape[0] - 1) - 1,
            ) / self.strides[0] as f64)
                + 1.0;
            let height = match self.ceil_mode {
                true => height.ceil(),
                false => height.floor(),
            };

            let width = (f64::from(
                x_shape[3] + 2 * self.pads[1] - self._dilations[1] * (self.kernel_shape[1] - 1) - 1,
            ) / self.strides[1] as f64)
                + 1.0;
            let width = match self.ceil_mode {
                true => width.ceil(),
                false => width.floor(),
            };

            let mut buf = self.alloc.alloc_with_value(0, 4);
            buf[0] = x_shape[0];
            buf[1] = self.kernel_shape[1];
            buf[2] = i32::from_f64(height).ok_or(Error::ComputationError)?;
            buf[3] = i32::from_f64(width).ok_or(Error::ComputationError)?;

            Ok(buf)
        } else if self.kernel_dims == 3 {
            // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d
            let depth = (f64::from(
                x_shape[1] + 2 * self.pads[0] - self._dilations[0] * (self.kernel_shape[0] - 1) - 1,
            ) / self.strides[0] as f64)
                + 1.0;
            let depth = match self.ceil_mode {
                true => depth.ceil(),
                false => depth.floor(),
            };

            let height = (f64::from(
                x_shape[2] + 2 * self.pads[1] - self._dilations[1] * (self.kernel_shape[1] - 1) - 1,
            ) / self.strides[1] as f64)
                + 1.0;
            let height = match self.ceil_mode {
                true => height.ceil(),
                false => height.floor(),
            };

            let width = (f64::from(
                x_shape[3] + 2 * self.pads[2] - self._dilations[2] * (self.kernel_shape[2] - 1) - 1,
            ) / self.strides[2] as f64)
                + 1.0;
            let width = match self.ceil_mode {
                true => width.ceil(),
                false => width.floor(),
            };

            let mut buf = self.alloc.alloc_with_value(0, 5);
            buf[0] = x_shape[0];
            buf[1] = self.kernel_shape[1];
            buf[2] = i32::from_f64(depth).ok_or(Error::ComputationError)?;
            buf[3] = i32::from_f64(height).ok_or(Error::ComputationError)?;
            buf[4] = i32::from_f64(width).ok_or(Error::ComputationError)?;

            Ok(buf)
        } else {
            unreachable!("Constructor method validates that only 2d and 3d are supported");
        }
    }
}

#[cfg(test)]
mod test {
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::CudaKernel;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils;
    use crate::test_utils::{TestMaxPoolAttributes, TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_max_pool_f32_2d() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![1, 1, 4, 4];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(CudaData::F32(
                device
                    .htod_copy(vec![
                        1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0,
                        4.0,
                    ])
                    .unwrap(),
            )),
        };

        let node_c = TestNode {
            shape: vec![1, 1, 2, 2],
            dtype,
            data: None,
        };

        let attributes = test_utils::create_max_pool_attributes(TestMaxPoolAttributes {
            dilations: None,
            kernel_shape: Some(Box::new([2, 2])),
            strides: Some(Box::new([2, 2])),
            row_major_order: None,
            ceil_mode: None,
            pads: None,
        });

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes,
            op: Op::MaxPool,
        };

        let (_, state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(state, 1).unwrap();

        let cuda_kernel = CudaKernel::new(Op::MaxPool, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 3.0, 4.0])
    }
}
