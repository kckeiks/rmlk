use crate::core::allocators::ScratchAllocator;
use crate::core::error::{InternalError, Result};
use log::{debug, warn};
use rmlk_schema::Attribute;
use std::collections::HashMap;

pub struct MaxPoolAttributes<'a> {
    pads: &'a [i32],
    strides: &'a [i32],
    kernel_shape: &'a [i32],
    _ceil_mode: bool,
    _row_major_order: bool,
    _dilations: &'a [i32],
}

impl<'a> MaxPoolAttributes<'a> {
    pub fn new(
        attrs: &'a HashMap<Box<str>, Attribute>,
        scratch_alloc: &'a ScratchAllocator,
    ) -> Result<Self> {
        let kernel_shape = match attrs.get("kernel_shape") {
            Some(attr) => attr
                .ints()
                .ok_or_else(|| InternalError::InvalidAttributeDataType {
                    name: "kernel_shape".to_string(),
                })?,
            None => {
                return Err(InternalError::MissingAttribute {
                    name: "kernel_shape".to_string(),
                })
            }
        };

        let pads;
        let strides;
        let mut ceil_mode = None;
        let mut row_major_order = None;

        if let Some(attr) = attrs.get("ceil_mode") {
            ceil_mode = match attr.int() {
                Some(0) => Some(false),
                Some(n) => {
                    debug!("unsupported attribute type `{n}` for `ceil_mode`");
                    return Err(InternalError::InvalidAttribute {
                        name: "ceil_mode".to_string(),
                    });
                }
                None => None,
            }
        }

        if attrs.get("dilations").is_some() {
            // Todo: how do we add support for this?
            warn!("unsupported attributes");
        }

        let kernel_dims = kernel_shape.len();

        if let Some(attr) = attrs.get("pads") {
            pads = attr
                .ints()
                .ok_or_else(|| InternalError::InvalidAttributeDataType {
                    name: "pads".to_string(),
                })?;
        } else {
            pads = scratch_alloc.allocate_fill(kernel_dims, 0)?;
        }

        if let Some(attr) = attrs.get("strides") {
            strides = attr
                .ints()
                .ok_or_else(|| InternalError::InvalidAttributeDataType {
                    name: "strides".to_string(),
                })?;
        } else {
            strides = scratch_alloc.allocate_fill(kernel_dims, 1)?;
        }

        if let Some(attr) = attrs.get("row_major_order") {
            row_major_order = match attr.int() {
                Some(0) => Some(true),
                _ => None,
            }
        }

        Ok(Self {
            pads,
            strides,
            kernel_shape,
            _dilations: scratch_alloc.allocate_fill(kernel_dims, 1)?,
            _ceil_mode: ceil_mode.unwrap_or(false),
            _row_major_order: row_major_order.unwrap_or(false),
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

    pub fn _ceil_mode(&self) -> bool {
        false
    }
}
