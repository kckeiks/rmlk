use std::alloc::Global;
use std::ptr::NonNull;
use crate::device;
use crate::device::{Device, DeviceError, EncodedTensor, Tensor};
use crate::node::Node;

#[derive(Clone)]
pub struct CpuDevice;

impl Device for CpuDevice {
    type Tensor = CpuTensor;
    type Allocator = Global;
    fn allocator(&self) -> Self::Allocator {
        Global
    }

    fn new_tensor(&self, input: EncodedTensor) -> device::Result<Self::Tensor> {
        input.try_into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuTensor {
    pub(crate) op: Option<()>,
    pub(crate) value: u32,
}

impl Tensor<CpuDevice> for CpuTensor {
    type Op = ();

    fn op(&self) -> Option<Self::Op> {
        self.op
    }

    fn forward(&mut self, inputs: &[NonNull<Node<CpuDevice>>]) -> device::Result<()> {
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

    fn try_from(value: EncodedTensor) -> device::Result<Self> {
        Ok(Self {
            op: if value.op { Some(()) } else { None },
            value: value.value,
        })
    }
}
