pub mod activation;
pub mod add;
pub mod conv;
pub mod gemm;
pub mod global_average_pool;
pub mod max_pool;

use crate::core::context::Context;
use crate::core::error::Result;
use crate::core::kernel::Kernel;
use crate::core::ops::flatten::FlattenOp;
use crate::core::provider::cuda::data::CudaData;
use crate::core::provider::cuda::kernel::activation::ActivationKernel;
use crate::core::provider::cuda::kernel::add::AddKernel;
use crate::core::provider::cuda::kernel::conv::ConvKernel;
use crate::core::provider::cuda::kernel::gemm::GemmKernel;
use crate::core::provider::cuda::kernel::global_average_pool::GlobalAveragePoolKernel;
use crate::core::provider::cuda::kernel::max_pool::MaxPoolKernel;

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
    type Data = CudaData;

    fn compute(self, ctx: &mut Context<Self::Data>) -> Result<()> {
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
