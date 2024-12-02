pub mod activation;
pub mod add;
pub mod conv;
pub mod gemm;
pub mod global_average_pool;
pub mod max_pool;

use crate::core::error::Result;
use crate::core::kernel::KernelBackend;
use crate::core::Context;
use crate::ops::activation::ActivationOp;
use crate::ops::add::AdditionOp;
use crate::ops::conv::ConvolutionOp;
use crate::ops::flatten::FlattenOp;
use crate::ops::gemm::GemmOp;
use crate::ops::global_average::GlobalAverageOp;
use crate::ops::max_pool::MaxPoolOp;
use crate::providers::cuda::Cuda;

pub enum CudaKernel {
    Add(AdditionOp<add::BackendHandler<add::ActiveKernel>>),
    Relu(ActivationOp<activation::BackendHandler<activation::ActiveKernel>>),
    Conv(ConvolutionOp<conv::BackendHandler<conv::ActiveKernel>>),
    Gemm(GemmOp<gemm::BackendHandler<gemm::ActiveKernel>>),
    GlobalAveragePool(
        GlobalAverageOp<global_average_pool::BackendHandler<global_average_pool::ActiveKernel>>,
    ),
    MaxPool(MaxPoolOp<max_pool::BackendHandler<max_pool::ActiveKernel>>),
    Flatten(FlattenOp), // Todo: How will we handle Flatten, for example?
}

impl KernelBackend for CudaKernel {
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

pub enum NoOpCudaKernel {
    Add(AdditionOp<add::BackendHandler<add::NoOpKernel>>),
    Relu(ActivationOp<activation::BackendHandler<activation::NoOpKernel>>),
    Conv(ConvolutionOp<conv::BackendHandler<conv::NoOpKernel>>),
    Gemm(GemmOp<gemm::BackendHandler<gemm::NoOpKernel>>),
    GlobalAveragePool(
        GlobalAverageOp<global_average_pool::BackendHandler<global_average_pool::NoOpKernel>>,
    ),
    MaxPool(MaxPoolOp<max_pool::BackendHandler<max_pool::NoOpKernel>>),
    Flatten(FlattenOp), // Todo: How will we handle Flatten, for example?
}

impl KernelBackend for NoOpCudaKernel {
    type Device = Cuda;

    fn compute(self, ctx: &mut Context<Self::Device>) -> Result<()> {
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
