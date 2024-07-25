use rmlk_ir::Op;
use crate::core::error::Error;

pub trait Kernel {
    fn compute(&self, op: Op) -> Result<(), Error>;
}

pub enum KernelComputer {
    Cuda,
    Cpu,
}