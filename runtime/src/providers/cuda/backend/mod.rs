pub mod activation;
pub mod add;
pub mod conv;
pub mod gemm;
pub mod global_average_pool;
pub mod max_pool;

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
use crate::providers::cuda::Cuda;

pub enum CudaKernel {
    Add(AdditionBackend<add::ActiveKernel>),
    Relu(ActivationBackend<activation::ActiveKernel>),
    Conv(ConvolutionBackend<conv::ActiveKernel>),
    Gemm(GemmBackend<gemm::ActiveKernel>),
    GlobalAveragePool(GlobalAverageBackend<global_average_pool::ActiveKernel>),
    MaxPool(MaxPoolBackend<max_pool::ActiveKernel>),
    Flatten(FlattenTemplate), // Todo: How will we handle Flatten, for example?
}

impl OperationBackend<Cuda> for CudaKernel {
    fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
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

#[allow(dead_code)]
pub enum NoOpCudaKernel {
    Add(AdditionBackend<add::NoOpKernel>),
    Relu(ActivationBackend<activation::NoOpKernel>),
    Conv(ConvolutionBackend<conv::NoOpKernel>),
    Gemm(GemmBackend<gemm::NoOpKernel>),
    GlobalAveragePool(GlobalAverageBackend<global_average_pool::NoOpKernel>),
    MaxPool(MaxPoolBackend<max_pool::NoOpKernel>),
    Flatten(FlattenTemplate), // Todo: How will we handle Flatten, for example?
}

impl OperationBackend<Cuda> for NoOpCudaKernel {
    fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        match self {
            NoOpCudaKernel::Add(kernel) => kernel.compute(ctx),
            NoOpCudaKernel::Relu(kernel) => kernel.compute(ctx),
            NoOpCudaKernel::Conv(kernel) => kernel.compute(ctx),
            NoOpCudaKernel::Gemm(kernel) => kernel.compute(ctx),
            NoOpCudaKernel::GlobalAveragePool(kernel) => kernel.compute(ctx),
            NoOpCudaKernel::MaxPool(kernel) => kernel.compute(ctx),
            NoOpCudaKernel::Flatten(kernel) => kernel.compute(ctx),
        }
    }
}
