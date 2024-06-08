use crate::node::Node;
use std::alloc::{Allocator, Global};
use std::ops::Add;
use std::ptr::NonNull;

#[derive(Debug)]
pub enum DeviceError {
    Unknown,
}

pub type Result<T> = std::result::Result<T, DeviceError>;

pub trait Device: Clone {
    type Tensor: TensorTr<Self> + TryFrom<EncodedTensor, Error = DeviceError>;
    type Allocator: Allocator + Clone;
    type Op;
    fn allocator(&self) -> Self::Allocator;

    fn new_tensor(&self, input: EncodedTensor) -> Result<Self::Tensor> {
        input.try_into()
    }
}

#[derive(Clone)]
pub struct CpuDevice;

impl Device for CpuDevice {
    type Tensor = MockTensor;
    type Allocator = Global;
    type Op = ();

    fn allocator(&self) -> Self::Allocator {
        Global
    }
}

pub struct EncodedTensor {
    pub op: bool,
    pub value: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MockTensor {
    pub(crate) value: Option<u32>,
}

impl Add for MockTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: Some(self.value.unwrap() + rhs.value.unwrap()),
        }
    }
}

pub trait TensorTr<D: Device> {
    fn execute(&mut self, inputs: &[NonNull<Node<D>>]) -> Result<()>;
}

impl TensorTr<CpuDevice> for MockTensor {
    fn execute(&mut self, inputs: &[NonNull<Node<CpuDevice>>]) -> Result<()> {
        let mut sum = 0;
        for input in inputs.iter().copied() {
            unsafe {
                *self = *self + *input.as_ref().tensor();
            }
        }
        Ok(())
    }
}

impl TryFrom<EncodedTensor> for MockTensor {
    type Error = DeviceError;

    fn try_from(value: EncodedTensor) -> Result<Self> {
        Ok(Self {
            value: Some(value.value),
        })
    }
}
