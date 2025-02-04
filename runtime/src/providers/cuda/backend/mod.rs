pub mod activation;
pub mod add;
mod binary;
pub mod conv;
mod gather;
pub mod gemm;
pub mod global_average_pool;
pub mod max_pool;
pub mod whereop;
mod common;

use crate::core::backend::OperationBackend;
use crate::core::error::Result;
use crate::core::Context;
use crate::providers::cpu::flatten::FlattenTemplate;
use crate::providers::cuda::activation::ActivationBackend;
use crate::providers::cuda::add::AdditionBackend;
use crate::providers::cuda::conv::ConvolutionBackend;
use crate::providers::cuda::gemm::GemmBackend;
use crate::providers::cuda::global_average_pool::GlobalAverageBackend;
use crate::providers::cuda::max_pool::MaxPoolBackend;
use crate::providers::cuda::whereop::WhereBackend;
use crate::providers::cuda::Cuda;

pub enum CudaKernel {
    Add(AdditionBackend),
    Relu(ActivationBackend),
    Conv(ConvolutionBackend),
    Gemm(GemmBackend),
    GlobalAveragePool(GlobalAverageBackend),
    MaxPool(MaxPoolBackend),
    Flatten(FlattenTemplate), // Todo: How will we handle Flatten, for example?
    Where(WhereBackend),
}

impl OperationBackend<Cuda> for CudaKernel {
    fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        match self {
            CudaKernel::Add(kernel) => kernel.compute::<add::ActiveKernel>(ctx),
            CudaKernel::Relu(kernel) => kernel.compute::<activation::ActiveKernel>(ctx),
            CudaKernel::Conv(kernel) => kernel.compute::<conv::ActiveKernel>(ctx),
            CudaKernel::Gemm(kernel) => kernel.compute::<gemm::ActiveKernel>(ctx),
            CudaKernel::GlobalAveragePool(kernel) => {
                kernel.compute::<global_average_pool::ActiveKernel>(ctx)
            }
            CudaKernel::MaxPool(kernel) => kernel.compute::<max_pool::ActiveKernel>(ctx),
            CudaKernel::Flatten(kernel) => kernel.compute(ctx),
            CudaKernel::Where(kernel) => kernel.compute::<whereop::ActiveKernel>(ctx),
        }
    }
}

impl CudaKernel {
    pub fn noop_compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        match self {
            CudaKernel::Add(kernel) => kernel.compute::<add::NoOpKernel>(ctx),
            CudaKernel::Relu(kernel) => kernel.compute::<activation::NoOpKernel>(ctx),
            CudaKernel::Conv(kernel) => kernel.compute::<conv::NoOpKernel>(ctx),
            CudaKernel::Gemm(kernel) => kernel.compute::<gemm::NoOpKernel>(ctx),
            CudaKernel::GlobalAveragePool(kernel) => {
                kernel.compute::<global_average_pool::NoOpKernel>(ctx)
            }
            CudaKernel::MaxPool(kernel) => kernel.compute::<max_pool::NoOpKernel>(ctx),
            CudaKernel::Flatten(kernel) => kernel.compute(ctx),
            CudaKernel::Where(kernel) => kernel.compute::<whereop::NoOpKernel>(ctx),
        }
    }
}
