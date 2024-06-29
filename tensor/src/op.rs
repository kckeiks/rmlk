#[derive(Debug)]
pub enum Op {
    Add,
    Mul,
}




pub struct OpKernelContext {
    pub start_index: Vec<>,
}

pub trait OpKernel {
    type Error;
    fn compute(&self, ctx: OpKernelContext) -> Result<(), Self::Error> ;
}