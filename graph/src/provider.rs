use rmlk_tensor::Op;
use crate::node::Link;

pub enum Provider {
    Cpu,
    Cuda,
    Metal,
}

impl Provider {
    pub fn compute<T>(&self, op: &Op, src: &[Option<Link<T>>]) {
        unimplemented!()
    }
}
