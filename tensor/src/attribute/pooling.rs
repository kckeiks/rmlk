use crate::kernel::Allocator;
use crate::Error;
use num_traits::FromPrimitive;
use rmlk_ir::Attribute;
use std::collections::HashMap;

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
    pub fn new(attrs: &HashMap<Box<str>, Attribute>, kernel_dims: usize) -> crate::Result<Self> {
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

    pub fn calculate_output_shape(&self, x_shape: &[i32]) -> crate::Result<Box<[i32]>> {
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

    pub fn _calculate_output_shape_pytorch(&self, x_shape: &[i32]) -> crate::Result<Box<[i32]>> {
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
