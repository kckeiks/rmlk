use crate::execution_state::ExecutionState;
use crate::tensor::Tensor;
use crate::Error;
use rmlk_ir::{Attribute, AttributeType};
use std::collections::HashMap;

pub trait Kernel {
    type Data;

    fn compute(&self, ctx: &mut Context<Self::Data>) -> crate::Result<()>;
}

pub struct Context<T> {
    execution_state: ExecutionState<T>,
    current_node: usize,
    input_count: usize,
}

impl<T> Context<T> {
    pub fn new(execution_state: ExecutionState<T>, node_id: usize) -> crate::Result<Self> {
        let input_count = execution_state
            .get_input_count(node_id)
            .ok_or(Error::MissingNodeInGraph)?;
        Ok(Self {
            execution_state,
            current_node: node_id,
            input_count,
        })
    }

    pub fn get_input(&self, index: usize) -> crate::Result<&Tensor<T>> {
        self.execution_state
            .get_tensor(self.current_node, index)
            .ok_or(Error::MissingTensor)
    }

    pub fn get_output(&mut self, index: usize) -> crate::Result<&Tensor<T>> {
        self.execution_state
            .get_tensor(
                self.current_node,
                self.input_count.checked_add(index).ok_or(Error::Overflow)?,
            )
            .ok_or(Error::MissingTensor)
    }

    pub fn get_output_mut(&mut self, index: usize) -> crate::Result<&mut Tensor<T>> {
        self.execution_state
            .get_tensor_mut(
                self.current_node,
                self.input_count.checked_add(index).ok_or(Error::Overflow)?,
            )
            .ok_or(Error::MissingTensor)
    }

    pub fn get_attributes(&self) -> Option<&HashMap<Box<str>, Attribute>> {
        Some(self.execution_state.get_node(self.current_node)?.attrs())
    }
}

pub struct OpKernelAttributes {
    inner: InnerOpKernelAttributes,
}

impl OpKernelAttributes {
    pub fn get_conv_attr(self) -> Result<ConvAttributes, crate::error::Error> {
        match self.inner {
            InnerOpKernelAttributes::Conv(attr) => Ok(attr),
        }
    }
}

pub enum InnerOpKernelAttributes {
    Conv(ConvAttributes),
}

// Maybe this is a good place to try small vec.
#[derive(Default)]
pub struct ConvAttributes {
    pub dilations: Option<Box<[i32]>>,
    pub group: Option<i32>,
    pub kernel_shape: Option<Box<[i32]>>,
    pub pads: Option<Box<[i32]>>,
    pub strides: Option<Box<[i32]>>,
}

impl TryFrom<&HashMap<Box<str>, Attribute>> for ConvAttributes {
    type Error = Error;

    fn try_from(value: &HashMap<Box<str>, Attribute>) -> Result<Self, Self::Error> {
        // Todo: We can probably do better than this
        let mut attrs = ConvAttributes {
            dilations: None,
            group: None,
            kernel_shape: None,
            pads: None,
            strides: None,
        };

        if let Some(attr) = value.get("dilations") {
            attrs.dilations = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }
        if let Some(attr) = value.get("group") {
            attrs.group = Some(attr.int().ok_or(Error::InvalidAttributeFormat)?);
        }
        if let Some(attr) = value.get("kernel_shape") {
            attrs.kernel_shape = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }
        if let Some(attr) = value.get("pads") {
            attrs.pads = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }
        if let Some(attr) = value.get("strides") {
            attrs.strides = Some(
                attr.ints()
                    .ok_or(Error::InvalidAttributeFormat)?
                    .to_vec()
                    .into_boxed_slice(),
            );
        }

        Ok(attrs)
    }
}
