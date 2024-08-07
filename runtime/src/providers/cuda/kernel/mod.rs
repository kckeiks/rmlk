pub mod activation;
pub mod add;
pub mod conv;
pub mod gemm;
pub mod global_average_pool;
pub mod max_pool;

use crate::core::Context;
use crate::core::Kernel;
use crate::core::Result;
use crate::ops::flatten::FlattenOp;
use crate::providers::cuda::kernel::activation::ActivationKernel;
use crate::providers::cuda::kernel::add::AddKernel;
use crate::providers::cuda::kernel::conv::ConvKernel;
use crate::providers::cuda::kernel::gemm::GemmKernel;
use crate::providers::cuda::kernel::global_average_pool::GlobalAveragePoolKernel;
use crate::providers::cuda::kernel::max_pool::MaxPoolKernel;
use crate::providers::cuda::CudaProvider;

pub enum CudaKernel {
    Add(AddKernel),
    Relu(ActivationKernel),
    Conv(ConvKernel),
    Gemm(GemmKernel),
    GlobalAveragePool(GlobalAveragePoolKernel),
    MaxPool(MaxPoolKernel),
    Flatten(FlattenOp), // Todo: How will we handle Flatten, for example?
}

impl Kernel for CudaKernel {
    type Provider = CudaProvider;

    fn compute(self, ctx: &mut Context<Self::Provider>) -> Result<()> {
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
