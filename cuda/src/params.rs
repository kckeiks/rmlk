use cudarc::cublas::sys;
use half::f16;

pub trait CudaParamMap {
    fn data_type() -> sys::cudaDataType_t;
    fn cublas_compute_type() -> sys::cublasComputeType_t;
    fn cublas_gemma_algo() -> sys::cublasGemmAlgo_t;
}

impl CudaParamMap for f32 {
    fn data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_32F
    }

    fn cublas_compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }

    fn cublas_gemma_algo() -> sys::cublasGemmAlgo_t {
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP
    }
}

impl CudaParamMap for f16 {
    fn data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_16F
    }

    fn cublas_compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_16F
    }

    fn cublas_gemma_algo() -> sys::cublasGemmAlgo_t {
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP
    }
}
