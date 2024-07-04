use crate::kernel::Kernel;
use rmlk_ir::Op;

pub trait Provider: Clone + Sized {
    type Kernel: Kernel;

    fn kernel(&self, op: Op) -> Self::Kernel;
}
