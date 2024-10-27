use crate::core::{Error, Result};
use log::warn;
use rmlk_ir::ir_v2::Attribute;
use std::collections::HashMap;

pub struct MaxPoolAttributes {
    _ceil_mode: bool,
    _dilations: Box<[i32]>,
    kernel_shape: Box<[i32]>,
    pads: Box<[i32]>,
    _row_major_order: bool,
    strides: Box<[i32]>,
}

impl MaxPoolAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>) -> Result<Self> {
        let kernel_shape = match attrs.get("kernel_shape") {
            Some(attr) => attr
                .ints()
                .ok_or(Error::InvalidAttributeFormat)?
                // Todo: remove the allocation here.
                .to_vec()
                .into_boxed_slice(),
            None => return Err(Error::MissingAttributes),
        };

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
            // Todo: how do we add support for this?
            warn!("unsupported attributes");
            // return Err(Error::UnsupportedAttribute);
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

        let kernel_dims = kernel_shape.len();

        Ok(Self {
            _dilations: vec![1; kernel_dims].into_boxed_slice(),
            _ceil_mode: ceil_mode.unwrap_or(false),
            pads: pads.unwrap_or_else(|| vec![0; kernel_dims].into_boxed_slice()),
            _row_major_order: row_major_order.unwrap_or(false),
            strides: strides.unwrap_or_else(|| vec![1; kernel_dims].into_boxed_slice()),
            kernel_shape,
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
