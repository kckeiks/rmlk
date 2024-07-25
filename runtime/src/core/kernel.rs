use crate::core::error::Error;
use crate::core::execution_state::ExecutionState;
use crate::core::session::SessionState;
use rmlk_ir::Op;
use rmlk_tensor::CudaData;

pub trait KernelComputer {
    fn compute(&self, op: Op) -> Result<(), Error>;
}

pub enum ProviderComputer {
    Cuda(CudaComputer),
    Cpu,
}

pub struct CudaComputer {
    execution_state: ExecutionState<CudaData>,
}

impl CudaComputer {
    pub fn new(session_state: &SessionState) -> Self {
        todo!()
    }
}

impl KernelComputer for CudaComputer {
    fn compute(&self, op: Op) -> Result<(), Error> {
        todo!()
    }
}
