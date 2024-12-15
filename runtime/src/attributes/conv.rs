use crate::core::allocators::ScratchAllocator;
use crate::core::error::{InternalError, Result};
use rmlk_schema::Attribute;
use std::collections::HashMap;

pub struct ConvAttributes<'a> {
    kernel_dims: usize,
    group: i32,
    dilations: &'a [i32],
    pads: &'a [i32],
    strides: &'a [i32],
    _kernel_shape: Option<&'a [i32]>,
}

impl<'a> ConvAttributes<'a> {
    // Todo: we use a scratch buffer to allocate some default slices
    // but instead we should allocate these buffers during deserialization of the model.
    pub fn new(
        attrs: &'a HashMap<Box<str>, Attribute>,
        scratch_alloc: &'a ScratchAllocator,
        kernel_dims: usize,
    ) -> Result<Self> {
        // Todo: We can probably do better than this
        let pads;
        let strides;
        let dilations;
        let mut group = None;
        let mut kernel_shape = None;

        if let Some(attr) = attrs.get("dilations") {
            dilations = attr
                .ints()
                .ok_or_else(|| InternalError::InvalidAttributeDataType {
                    name: "dilations".to_string(),
                })?;
        } else {
            dilations = scratch_alloc.allocate_fill(kernel_dims, 1)?;
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
            kernel_shape =
                Some(
                    attr.ints()
                        .ok_or_else(|| InternalError::InvalidAttributeDataType {
                            name: "kernel_shape".to_string(),
                        })?,
                );
        }

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

        Ok(Self {
            dilations,
            group: group.unwrap_or(1),
            _kernel_shape: kernel_shape,
            pads,
            strides,
            kernel_dims,
        })
    }

    pub fn pads(&self) -> &[i32] {
        debug_assert!(self.kernel_dims == 2 || self.kernel_dims == 3);
        self.pads[2..].as_ref()
    }

    pub fn dilations(&self) -> &[i32] {
        self.dilations
    }

    pub fn strides(&self) -> &[i32] {
        self.strides
    }

    pub fn group(&self) -> i32 {
        self.group
    }

    pub fn _kernel_shape(&self) -> Option<&[i32]> {
        self._kernel_shape
    }
}
