use std::ptr::NonNull;
use rmlk_tensor::Op;
use crate::tensor::Tensor;

pub type Link<T> = Option<NonNull<Node<T>>>;

pub struct Node<T> {
    tensor: T,
    op: Op,
    src: [Link<T>; 4],
}

impl<T> Node<T>
where
    T: Tensor
{
    pub fn tensor(&self) -> &T {
        todo!()
    }

    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn src(&self) -> &[Link<T>] {
        todo!()
    }
}
