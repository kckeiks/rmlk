use crate::context::ExecutionContext;
use crate::cuda::CudaProvider;
use crate::kernel::Kernel;
use crate::Result;
use rmlk_ir::Op;
pub struct CudaKernel(Op);

impl From<Op> for CudaKernel {
    fn from(value: Op) -> Self {
        Self(value)
    }
}

impl Kernel for CudaKernel {
    type Provider = CudaProvider;

    fn compute(&self, ctx: &mut ExecutionContext<Self::Provider>) -> Result<()> {
        match self.0 {
            Op::Add => {}
            Op::MatMul => {}
            Op::Gemm => {}
            Op::Conv => {}
            _ => todo!(),
        }

        Ok(())
    }
}
