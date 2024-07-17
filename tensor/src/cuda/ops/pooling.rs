use std::collections::HashMap;
use crate::cuda::data::CudaData;
use crate::{Context, Error, Result};
use cudarc::cudnn::{Cudnn};
use cudarc::driver::{CudaDevice};

use std::sync::Arc;
use rmlk_ir::Attribute;
use crate::kernel::Allocator;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    // Input data tensor.
    let x = ctx.get_input(0)?;
    // Todo: Let's define an attributes object.

    // Todo: Finish.
    Ok(())
}

pub struct MaxPoolAttributes {
    ceil_mode: bool,
    dilations: Box<[i32]>,
    kernel_shape: Box<[i32]>,
    pads: Box<[i32]>,
    row_major_order: bool,
    strides: Box<[i32]>,
    kernel_dims: usize,
    alloc: Allocator,
}

impl MaxPoolAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>, kernel_dims: usize) -> Result<Self> {
        let alloc = Allocator;
        let mut dilations = None;
        let mut ceil_mode = None;
        let mut pads = None;
        let mut strides = None;
        let mut row_major_order = None;

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

        if let Some(attr) = attrs.get("ceil_mode") {
            ceil_mode = match attr.int() {
                Some(0) => Some(false),
                _ => None,
            }
        }

        if let Some(attr) = attrs.get("dilations") {
            dilations = Some(
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

        if let Some(attr) = attrs.get("row_major_order") {
            row_major_order = match attr.int() {
                Some(0) => Some(true),
                _ => None,
            }
        }

        Ok(Self {
            dilations: dilations.unwrap_or_else(|| alloc.alloc_with_value::<i32>(1, kernel_dims)),
            ceil_mode: ceil_mode.unwrap_or(false),
            kernel_shape: kernel_shape.ok_or(Error::MissingAttributes)?,
            pads: pads.unwrap_or_else(|| alloc.alloc_with_value::<i32>(0, kernel_dims)),
            row_major_order: row_major_order.unwrap_or(false),
            strides: strides.unwrap_or_else(|| alloc.alloc_with_value::<i32>(1, kernel_dims)),
            kernel_dims,
            alloc,
        })
    }

    pub fn pads(&self) -> &[i32] {
        self.pads.as_ref()
    }

    pub fn dilations(&self) -> &[i32] {
        self.dilations.as_ref()
    }

    pub fn strides(&self) -> &[i32] {
        self.pads.as_ref()
    }

    pub fn kernel_shape(&self) -> &[i32] {
        self.kernel_shape.as_ref()
    }

    pub fn calculate_output_shape(&self, x_shape: &[i32], kernel_shape: &[i32]) -> Box<[i32]> {
        if self.kernel_dims == 2 {
            // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d.
            let height = (f64::from(x_shape[2] + 2 * self.pads[0] - self.dilations[0] * (kernel_shape[0] - 1) - 1) / self.strides[0] as f64) + 1.0;
            let height = match self.ceil_mode {
                true => height.ceil(),
                false => height.floor(),
            };

            let width = (f64::from(x_shape[3] + 2 * self.pads[1] - self.dilations[1] * (kernel_shape[1] - 1) - 1) / self.strides[1] as f64) + 1.0;
            let width = match self.ceil_mode {
                true => width.ceil(),
                false => width.floor(),
            };

            let mut buf = self.alloc.alloc_with_value(0, 4);
            buf[0] = x_shape[0];
            buf[1] = kernel_shape[1];
            buf[2] = i32::from_f64(height).ok_or(Error::ComputationError)?;
            buf[3] = i32::from_f64(width).ok_or(Error::ComputationError)?;

            buf
        } else if self.kernel_dims == 3 {
            // For reference, see https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d
            let depth = (f64::from(x_shape[1] + 2 * self.pads[0] - self.dilations[0] * (kernel_shape[0] - 1) - 1) / self.strides[0] as f64) + 1.0;
            let depth = match self.ceil_mode {
                true => depth.ceil(),
                false => depth.floor(),
            };

            let height = (f64::from(x_shape[2] + 2 * self.pads[1] - self.dilations[1] * (kernel_shape[1] - 1) - 1) / self.strides[1] as f64) + 1.0;
            let height = match self.ceil_mode {
                true => height.ceil(),
                false => height.floor(),
            };

            let width = (f64::from(x_shape[3] + 2 * self.pads[2] - self.dilations[2] * (kernel_shape[2] - 1) - 1) / self.strides[2] as f64) + 1.0;
            let width = match self.ceil_mode {
                true => width.ceil(),
                false => width.floor(),
            };

            let mut buf = self.alloc.alloc_with_value(0, 5);
            buf[0] = x_shape[0];
            buf[1] = kernel_shape[1];
            buf[2] = i32::from_f64(depth).ok_or(Error::ComputationError)?;
            buf[3] = i32::from_f64(height).ok_or(Error::ComputationError)?;
            buf[4] = i32::from_f64(width).ok_or(Error::ComputationError)?;

            buf
        } else {
            unreachable!("Constructor method validates that only 2d and 3d are supported");
        }
    }
}