use crate::cuda::data::CudaData;
use crate::kernel::Context;
use crate::Error;
use crate::Result;
use cudarc::cublas::{sys, CudaBlas, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DevicePtr, DevicePtrMut};
use rmlk_ir::DataType;
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let lhs = ctx.get_input(0)?;
    let rhs = ctx.get_input(1)?;

    let lhs_shape = lhs.shape();
    let b = lhs_shape[..lhs_shape.len() - 2].iter().product::<usize>();
    let m = lhs_shape[lhs_shape.len() - 2];
    let k = lhs_shape[lhs_shape.len() - 1];

    let rhs_shape = rhs.shape();
    let n = rhs_shape[rhs_shape.len() - 2];

    match *lhs.dtype() {
        DataType::Float16 => {
            todo!()
        }
        DataType::Float => {
            let lhs_stride = lhs.stride();
            let rhs_stride = rhs.stride();
            let config = gemm_config::<f32>(
                1.0,
                0.0,
                (b, m, n, k),
                (lhs_shape, lhs_stride),
                (rhs_shape, rhs_stride),
            )
            .unwrap();

            // let out = ctx.allocate(DataType::Float, vec![b, m, n])?;
            let mut out_slice = unsafe { device.alloc::<f32>(b * m * n).unwrap() };

            let cublas = CudaBlas::new(device).unwrap();

            unsafe {
                gemm_stride_batched_f32(
                    &cublas,
                    config,
                    &rhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                    &lhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                    &mut out_slice,
                )
                .unwrap();
            };

            let out = ctx.get_output_mut(0)?;
            let _ = out.init(CudaData::F32(out_slice));
        }
        DataType::Double => {
            todo!()
        }
        _ => todo!(),
    }
    Ok(())
}

pub fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    // Todo: Make Layout object.
    // (shape, stride)
    lhs_layout: (&[usize], &[usize]),
    rhs_layout: (&[usize], &[usize]),
) -> Result<StridedBatchedConfig<T>> {
    let rhs_stride = rhs_layout.1;
    let (transa, lda) = match rhs_stride {
        [.., k_stride, 1] | [k_stride, 1] if *k_stride == k => {
            (sys::cublasOperation_t::CUBLAS_OP_N, n)
        }
        [.., 1, k_stride] | [1, k_stride] if *k_stride == k => {
            (sys::cublasOperation_t::CUBLAS_OP_T, k)
        }
        // Todo: return an non-contiguous error.
        _ => return Err(Error::Unknown),
    };

    let lhs_stride = lhs_layout.1;
    let (transb, ldb) = match lhs_stride {
        [.., m_stride, 1] | [m_stride, 1] if *m_stride == m => {
            (sys::cublasOperation_t::CUBLAS_OP_N, k)
        }
        [.., 1, m_stride] | [1, m_stride] if *m_stride == m => {
            (sys::cublasOperation_t::CUBLAS_OP_T, m)
        }
        // Todo: return an non-contiguous error.
        _ => return Err(Error::Unknown),
    };

    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda: lda as i32,
        ldb: ldb as i32,
        ldc: n as i32,
        transa,
        transb,
    };

    Ok(StridedBatchedConfig {
        gemm,
        batch_size: b as i32,
        stride_a: (n * k) as i64,
        stride_b: (k * m) as i64,
        stride_c: (m * n) as i64,
    })
}

pub unsafe fn gemm_stride_batched_f32(
    cublas: &CudaBlas,
    config: StridedBatchedConfig<f32>,
    a: &CudaView<f32>,
    b: &CudaView<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<()> {
    let alpha = &config.gemm.alpha as *const f32 as *const _;
    let beta = &config.gemm.beta as *const f32 as *const _;

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        config.gemm.transa,
        config.gemm.transb,
        config.gemm.m,
        config.gemm.n,
        config.gemm.k,
        alpha,
        *a.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        config.gemm.lda,
        config.stride_a,
        *b.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_32F,
        config.gemm.ldb,
        config.stride_b,
        beta,
        *c.device_ptr_mut() as *mut _,
        sys::cudaDataType_t::CUDA_R_32F,
        config.gemm.ldc,
        config.stride_c,
        config.batch_size,
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    .map_err(|_| Error::Unknown)
}
