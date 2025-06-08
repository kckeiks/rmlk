pub mod activation;
pub mod add;
mod binary;
pub mod cast;
mod common;
pub mod concat;
pub mod constant;
pub mod constant_of_shape;
pub mod conv;
pub mod gather;
pub mod gemm;
pub mod global_average_pool;
pub mod max_pool;
pub mod range;
pub mod reduce_mean;
mod relu;
pub mod shape;
pub mod sigmoid;
pub mod slice;
pub mod sqrt;
pub mod transpose;
mod trilu;
mod unary;
pub mod unsqueeze;
pub mod whereop;

use crate::core::backend::OperationBackend;
use crate::core::error::Result;
use crate::core::Context;
use crate::providers::cpu::flatten::FlattenTemplate;
use crate::providers::cuda::activation::ActivationBackend;
use crate::providers::cuda::add::AdditionBackend;
use crate::providers::cuda::backend::cast::CastBackend;
use crate::providers::cuda::backend::constant_of_shape::ConstantOfShapeBackend;
use crate::providers::cuda::backend::gather::GatherBackend;
use crate::providers::cuda::backend::shape::ShapeBackend;
use crate::providers::cuda::backend::sqrt::SqrtBackend;
use crate::providers::cuda::backend::transpose::TransposeBackend;
use crate::providers::cuda::concat::ConcatBackend;
use crate::providers::cuda::constant::ConstantBackend;
use crate::providers::cuda::conv::ConvolutionBackend;
use crate::providers::cuda::gemm::GemmBackend;
use crate::providers::cuda::global_average_pool::GlobalAverageBackend;
use crate::providers::cuda::max_pool::MaxPoolBackend;
use crate::providers::cuda::range::RangeBackend;
use crate::providers::cuda::reduce_mean::ReduceMeanBackend;
use crate::providers::cuda::sigmoid::SigmoidKernel;
use crate::providers::cuda::slice::SliceBackend;
use crate::providers::cuda::unsqueeze::UnsqueezeBackend;
use crate::providers::cuda::whereop::WhereBackend;
use crate::providers::cuda::Cuda;

pub enum CudaKernel {
    Add(AdditionBackend),
    Relu(ActivationBackend),
    Cast(CastBackend),
    Concat(ConcatBackend),
    Constant(ConstantBackend),
    Conv(ConvolutionBackend),
    ConstantOfShape(ConstantOfShapeBackend),
    Gather(GatherBackend),
    Gemm(GemmBackend),
    GlobalAveragePool(GlobalAverageBackend),
    MaxPool(MaxPoolBackend),
    Flatten(FlattenTemplate), // Todo: How will we handle Flatten, for example?
    Range(RangeBackend),
    ReduceMean(ReduceMeanBackend),
    Shape(ShapeBackend),
    Sigmoid(ActivationBackend),
    Slice(SliceBackend),
    Sqrt(SqrtBackend),
    Transpose(TransposeBackend),
    Unsqueeze(UnsqueezeBackend),
    Where(WhereBackend),
}

impl OperationBackend<Cuda> for CudaKernel {
    fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        match self {
            CudaKernel::Add(kernel) => kernel.compute::<add::ActiveKernel>(ctx),
            CudaKernel::Relu(kernel) => kernel.compute::<relu::ReluKernel>(ctx),
            CudaKernel::Cast(kernel) => kernel.compute::<cast::ActiveKernel>(ctx),
            CudaKernel::Concat(kernel) => kernel.compute(ctx),
            CudaKernel::Constant(kernel) => kernel.compute(ctx),
            CudaKernel::Conv(kernel) => kernel.compute::<conv::ActiveKernel>(ctx),
            CudaKernel::ConstantOfShape(backend) => backend.compute(ctx),
            CudaKernel::Gather(kernel) => kernel.compute::<gather::DefaultGatherProcessor>(ctx),
            CudaKernel::Gemm(kernel) => kernel.compute::<gemm::ActiveKernel>(ctx),
            CudaKernel::GlobalAveragePool(kernel) => {
                kernel.compute::<global_average_pool::ActiveKernel>(ctx)
            }
            CudaKernel::MaxPool(kernel) => kernel.compute::<max_pool::ActiveKernel>(ctx),
            CudaKernel::Flatten(kernel) => kernel.compute(ctx),
            CudaKernel::Range(backend) => backend.compute(ctx),
            CudaKernel::ReduceMean(backend) => backend.compute::<reduce_mean::ActiveKernel>(ctx),
            CudaKernel::Shape(kernel) => kernel.compute::<shape::DefaultShapeProcessor>(ctx),
            CudaKernel::Sigmoid(kernel) => kernel.compute::<SigmoidKernel>(ctx),
            CudaKernel::Slice(kernel) => kernel.compute(ctx),
            CudaKernel::Sqrt(kernel) => kernel.compute::<sqrt::ActiveKernel>(ctx),
            CudaKernel::Transpose(backend) => backend.compute(ctx),
            CudaKernel::Unsqueeze(backend) => backend.compute(ctx),
            CudaKernel::Where(kernel) => kernel.compute::<whereop::ActiveKernel>(ctx),
        }
    }
}

impl CudaKernel {
    pub fn noop_compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        match self {
            CudaKernel::Add(kernel) => kernel.compute::<add::NoOpKernel>(ctx),
            CudaKernel::Relu(kernel) => kernel.compute::<relu::NoOpKernel>(ctx),
            CudaKernel::Conv(kernel) => kernel.compute::<conv::NoOpKernel>(ctx),
            CudaKernel::ConstantOfShape(_) => {
                // Todo: Some operations complicate things for the noop computations because
                // the size of the output tensor's data depends on the input. Thus, we can't
                // pre-allocate memory.
                Ok(())
            }
            CudaKernel::Gather(kernel) => kernel.compute::<gather::NoOpGatherProcessor>(ctx),
            CudaKernel::Gemm(kernel) => kernel.compute::<gemm::NoOpKernel>(ctx),
            CudaKernel::GlobalAveragePool(kernel) => {
                kernel.compute::<global_average_pool::NoOpKernel>(ctx)
            }
            CudaKernel::MaxPool(kernel) => kernel.compute::<max_pool::NoOpKernel>(ctx),
            CudaKernel::Flatten(kernel) => kernel.compute(ctx),
            CudaKernel::Shape(kernel) => kernel.compute::<shape::NoOpShapeProcessor>(ctx),
            CudaKernel::Where(kernel) => kernel.compute::<whereop::NoOpKernel>(ctx),
            _ => unimplemented!(),
        }
    }
}
