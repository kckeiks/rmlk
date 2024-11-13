pub mod activation;
pub mod add;
pub mod conv;
pub mod gemm;
pub mod global_average_pool;
pub mod max_pool;

use crate::core::kernel::{Kernel, KernelError, Result};
use crate::core::Context;
use crate::ops::activation::ActivationOp;
use crate::ops::flatten::FlattenOp;
use crate::providers::cuda::kernel::add::AddKernel;
use crate::providers::cuda::kernel::conv::ConvKernel;
use crate::providers::cuda::kernel::gemm::GemmKernel;
use crate::providers::cuda::kernel::global_average_pool::GlobalAveragePoolKernel;
use crate::providers::cuda::kernel::max_pool::MaxPoolKernel;
use crate::providers::cuda::Cuda;

pub enum CudaKernel {
    Add(AddKernel),
    Relu(ActivationOp<activation::ActivationKernel>),
    Conv(ConvKernel),
    Gemm(GemmKernel),
    GlobalAveragePool(GlobalAveragePoolKernel),
    MaxPool(MaxPoolKernel),
    Flatten(FlattenOp), // Todo: How will we handle Flatten, for example?
}

impl Kernel for CudaKernel {
    type Device = Cuda;

    fn compute(self, ctx: &mut Context<Self::Device>) -> Result<()> {
        match self {
            CudaKernel::Add(kernel) => kernel.compute(ctx),
            CudaKernel::Relu(kernel) => kernel.compute(ctx),
            CudaKernel::Conv(kernel) => kernel.compute(ctx),
            CudaKernel::Gemm(kernel) => kernel.compute(ctx),
            CudaKernel::GlobalAveragePool(kernel) => kernel.compute(ctx),
            CudaKernel::MaxPool(kernel) => kernel.compute(ctx),
            CudaKernel::Flatten(kernel) => kernel.compute(ctx),
        }
    }
}

impl From<rmlk_cuda::Error> for KernelError {
    fn from(value: rmlk_cuda::Error) -> Self {
        Self::Device(value.into())
    }
}
