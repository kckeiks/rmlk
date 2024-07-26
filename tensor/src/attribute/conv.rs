use crate::kernel::Allocator;
use crate::Error;
use rmlk_ir::Attribute;
use std::collections::HashMap;

// Maybe this is a good place to try small vec.
pub struct ConvAttributes {
    pub dilations: Box<[i32]>,
    pub group: i32,
    pub kernel_shape: Option<Box<[i32]>>,
    pub pads: Box<[i32]>,
    pub strides: Box<[i32]>,
    kernel_dims: usize,
    alloc: Allocator,
}

impl ConvAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>, kernel_dims: usize) -> crate::Result<Self> {
        let alloc = Allocator;
        // Todo: We can probably do better than this
        let mut dilations = None;
        let mut group = None;
        let mut kernel_shape = None;
        let mut pads = None;
        let mut strides = None;

        if let Some(attr) = attrs.get("dilations") {
            dilations = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        if let Some(attr) = attrs.get("group") {
            group = Some(attr.int().ok_or(Error::InvalidAttributeFormat)?);
        }

        if let Some(attr) = attrs.get("kernel_shape") {
            kernel_shape = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
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

        // Todo: Validate input using kernel_dims.

        Ok(Self {
            dilations: dilations.unwrap_or_else(|| alloc.alloc_with_value::<i32>(1, kernel_dims)),
            group: group.unwrap_or(1),
            kernel_shape,
            pads: pads.unwrap_or_else(|| alloc.alloc_with_value::<i32>(0, kernel_dims)),
            strides: strides.unwrap_or_else(|| alloc.alloc_with_value::<i32>(1, kernel_dims)),
            kernel_dims,
            alloc,
        })
    }

    pub fn pads(&self) -> &[i32] {
        debug_assert!(self.kernel_dims == 2 || self.kernel_dims == 3);
        self.pads[2..].as_ref()
    }

    pub fn dilations(&self) -> &[i32] {
        self.dilations.as_ref()
    }

    pub fn strides(&self) -> &[i32] {
        self.pads.as_ref()
    }

    pub fn group(&self) -> i32 {
        self.group
    }

    // Todo: pass the output buffer.
    pub fn kernel_shape(&self, x_shape: &[i32]) -> Option<Box<[i32]>> {
        if let Some(shape) = self.kernel_shape.as_ref() {
            if self.kernel_dims == 2 {
                let mut buf = self.alloc.alloc_with_value(0, 4);

                buf[0] = x_shape[0];
                buf[1] = x_shape[1] / self.group;
                buf[2] = shape[0];
                buf[3] = shape[1];

                Some(buf)
            } else if self.kernel_dims == 3 {
                let mut buf = self.alloc.alloc_with_value(0, 5);

                buf[0] = x_shape[0];
                buf[1] = x_shape[1] / self.group;
                buf[2] = shape[0];
                buf[3] = shape[1];
                buf[3] = shape[1];

                Some(buf)
            } else {
                unreachable!("Constructor method validates that only 2d and 3d are supported");
            }
        } else {
            None
        }
    }

    pub fn calculate_output_shape(&self, x_shape: &[i32], w_shape: &[i32]) -> Box<[i32]> {
        if self.kernel_dims == 2 {
            // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d.
            let height =
                ((x_shape[2] + 2 * self.pads[0] - self.dilations[0] * (w_shape[2] - 1) - 1)
                    / self.strides[0])
                    + 1;
            let width =
                ((x_shape[3] + 2 * self.pads[1] - self.dilations[1] * (w_shape[3] - 1) - 1)
                    / self.strides[1])
                    + 1;

            let mut buf = self.alloc.alloc_with_value(0, 4);
            buf[0] = x_shape[0];
            buf[1] = w_shape[1];
            buf[2] = height;
            buf[3] = width;

            buf
        } else if self.kernel_dims == 3 {
            // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d.
            let depth =
                ((x_shape[0] + 2 * self.pads[0] - self.dilations[0] * (w_shape[2] - 1) - 1)
                    / self.strides[0])
                    + 1;
            let height =
                ((x_shape[2] + 2 * self.pads[1] - self.dilations[1] * (w_shape[3] - 1) - 1)
                    / self.strides[1])
                    + 1;
            let width =
                ((x_shape[3] + 2 * self.pads[2] - self.dilations[2] * (w_shape[4] - 1) - 1)
                    / self.strides[2])
                    + 1;

            let mut buf = self.alloc.alloc_with_value(0, 5);
            buf[0] = x_shape[0];
            buf[1] = w_shape[1];
            buf[2] = depth;
            buf[3] = height;
            buf[4] = width;

            buf
        } else {
            unreachable!("Constructor method validates that only 2d and 3d are supported");
        }
    }
}
