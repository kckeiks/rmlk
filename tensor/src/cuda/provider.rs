use crate::cuda::kernel::CudaKernel;
use crate::provider::Provider;
use cudarc::cudnn;
use cudarc::driver::CudaDevice;
use rmlk_ir::Op;
use std::sync::Arc;

pub struct CudaProvider {
    device: Arc<CudaDevice>,
    _cudnn: Arc<cudnn::Cudnn>,
}

impl Clone for CudaProvider {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            // Todo: We should be careful here.
            // Two threads cannot use a handle simultaneously,
            // See https://docs.nvidia.com/deeplearning/cudnn/latest/developer/misc.html?highlight=thread%20safety#thread-safety.
            _cudnn: self._cudnn.clone(),
        }
    }
}

impl CudaProvider {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        // Todo: Handle the unwrap.
        Self {
            _cudnn: cudnn::Cudnn::new(device.clone()).unwrap(),
            device,
        }
    }
}

impl Provider for CudaProvider {
    type Kernel = CudaKernel;

    fn kernel(&self, op: Op) -> Self::Kernel {
        CudaKernel::new(op, self.device.clone())
    }
}
