mod activation;
mod add;
mod conv;
mod gemm;
mod global_average_pool;
mod max_pool;

use crate::core::provider::cuda::kernel::activation::ActivationKernel;
use crate::core::provider::cuda::kernel::add::AddKernel;
use crate::core::provider::cuda::kernel::conv::ConvKernel;
use crate::core::provider::cuda::kernel::gemm::GemmKernel;
use crate::core::provider::cuda::kernel::global_average_pool::GlobalAveragePoolKernel;
use crate::core::provider::cuda::kernel::max_pool::MaxPoolKernel;

pub enum CudaKernel {
    Add(AddKernel),
    Activation(ActivationKernel),
    Conv(ConvKernel),
    Gemm(GemmKernel),
    GlobalAveragePool(GlobalAveragePoolKernel),
    MaxPool(MaxPoolKernel),
}
