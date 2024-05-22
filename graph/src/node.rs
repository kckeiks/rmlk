use std::ptr::NonNull;
use rmlk_tensor::device::Device;
use rmlk_tensor::op::Op;
use rmlk_tensor::tensor::Tensor;

pub type Link<D> = Option<NonNull<Node<D>>>;

pub struct Node<D: Device> {
    tensor: Tensor<D>,
    op: Op,
    src: [Link<D>; 4],
}

impl<D> Node<D>
where
    D: Device,
{
    pub fn tensor(&self) -> &D {
        todo!()
    }

    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn src(&self) -> &[Link<D>] {
        todo!()
    }
}
