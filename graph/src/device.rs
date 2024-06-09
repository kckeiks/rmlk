use crate::node::Node;
use std::alloc::{Allocator, Global};
use std::ptr::NonNull;

#[derive(Debug)]
pub enum DeviceError {
    Unknown,
}

pub type Result<T> = std::result::Result<T, DeviceError>;

pub trait Device: Clone {
    type Tensor: Tensor<Self>;
    type Allocator: Allocator + Clone;
    fn allocator(&self) -> Self::Allocator;
    fn new_tensor(&self, input: EncodedTensor) -> Result<Self::Tensor>;
}

#[derive(Clone)]
pub struct CpuDevice;

impl Device for CpuDevice {
    type Tensor = CpuTensor;
    type Allocator = Global;
    fn allocator(&self) -> Self::Allocator {
        Global
    }

    fn new_tensor(&self, input: EncodedTensor) -> Result<Self::Tensor> {
        input.try_into()
    }
}

pub struct EncodedTensor {
    pub op: bool,
    pub value: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuTensor {
    pub(crate) op: Option<()>,
    pub(crate) value: u32,
}

pub trait Tensor<D: Device> {
    type Op;
    fn op(&self) -> Option<Self::Op>;
    fn forward(&mut self, inputs: &[NonNull<Node<D>>]) -> Result<()>;
}

impl Tensor<CpuDevice> for CpuTensor {
    type Op = ();

    fn op(&self) -> Option<Self::Op> {
        self.op
    }

    fn forward(&mut self, inputs: &[NonNull<Node<CpuDevice>>]) -> Result<()> {
        let a = inputs.get(0).unwrap();
        let b = inputs.get(1).unwrap();
        let tensor_a = unsafe { a.as_ref().tensor() };
        let tensor_b = unsafe { b.as_ref().tensor() };
        self.value = tensor_a.value + tensor_b.value;
        Ok(())
    }
}

impl TryFrom<EncodedTensor> for CpuTensor {
    type Error = DeviceError;

    fn try_from(value: EncodedTensor) -> Result<Self> {
        Ok(Self {
            op: if value.op { Some(()) } else { None },
            value: value.value,
        })
    }
}
