use crate::core::error::{InternalError, Result};
use rmlk_schema::Attribute;
use std::collections::HashMap;

pub struct ConvAttributes {
    dilations: Box<[i32]>,
    group: i32,
    _kernel_shape: Option<Box<[i32]>>,
    pads: Box<[i32]>,
    strides: Box<[i32]>,
    kernel_dims: usize,
}

impl ConvAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>, kernel_dims: usize) -> Result<Self> {
        // Todo: We can probably do better than this
        let mut dilations = None;
        let mut group = None;
        let mut kernel_shape = None;
        let mut pads = None;
        let mut strides = None;

        if let Some(attr) = attrs.get("dilations") {
            dilations = Some(
                attr.ints()
                    .ok_or_else(|| InternalError::InvalidAttributeDataType {
                        name: "dilations".to_string(),
                    })?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        if let Some(attr) = attrs.get("group") {
            group = Some(
                attr.int()
                    .ok_or_else(|| InternalError::InvalidAttributeDataType {
                        name: "group".to_string(),
                    })?,
            );
        }

        if let Some(attr) = attrs.get("kernel_shape") {
            kernel_shape = Some(
                attr.ints()
                    .ok_or_else(|| InternalError::InvalidAttributeDataType {
                        name: "kernel_shape".to_string(),
                    })?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        if let Some(attr) = attrs.get("pads") {
            pads = Some(
                attr.ints()
                    .ok_or_else(|| InternalError::InvalidAttributeDataType {
                        name: "pads".to_string(),
                    })?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        if let Some(attr) = attrs.get("strides") {
            strides = Some(
                attr.ints()
                    .ok_or_else(|| InternalError::InvalidAttributeDataType {
                        name: "strides".to_string(),
                    })?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        Ok(Self {
            dilations: dilations.unwrap_or_else(|| vec![1; kernel_dims].into_boxed_slice()),
            group: group.unwrap_or(1),
            _kernel_shape: kernel_shape,
            pads: pads.unwrap_or_else(|| vec![0; kernel_dims].into_boxed_slice()),
            strides: strides.unwrap_or_else(|| vec![1; kernel_dims].into_boxed_slice()),
            kernel_dims,
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
        self.strides.as_ref()
    }

    pub fn group(&self) -> i32 {
        self.group
    }

    pub fn _kernel_shape(&self) -> Option<&Box<[i32]>> {
        self._kernel_shape.as_ref()
    }
}
