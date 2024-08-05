use crate::error::Error;
use crate::error::Result;
use cudarc::cublas::{sys, CudaBlas, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DevicePtr, DevicePtrMut};
use half::f16;
use log::trace;
use std::sync::Arc;

pub struct GemmOp {
    lhs_shape: [usize; 2],
    lhs_stride: [usize; 2],
    rhs_shape: [usize; 2],
    rhs_stride: [usize; 2],
    b: usize,
    m: usize,
    n: usize,
    k: usize,
}

impl GemmOp {
    pub fn new(
        lhs_shape: &[usize],
        lhs_stride: &[usize],
        rhs_shape: &[usize],
        rhs_stride: &[usize],
        trans_a: bool,
        trans_b: bool,
    ) -> Self {
        let b = lhs_shape[..lhs_shape.len() - 2].iter().product::<usize>();
        let (m, k) = match trans_a {
            true => {
                let m = lhs_shape[lhs_shape.len() - 1];
                let k = lhs_shape[lhs_shape.len() - 2];
                (m, k)
            }
            false => {
                let m = lhs_shape[lhs_shape.len() - 2];
                let k = lhs_shape[lhs_shape.len() - 1];
                (m, k)
            }
        };

        let n = match trans_b {
            true => rhs_shape[rhs_shape.len() - 1],
            false => rhs_shape[rhs_shape.len() - 2],
        };

        let lhs_dims = lhs_shape.len();
        let (lhs_shape, lhs_stride) = match trans_a {
            true => (
                [lhs_shape[lhs_dims - 1], lhs_shape[lhs_dims - 2]],
                [lhs_stride[lhs_dims - 1], lhs_stride[lhs_dims - 2]],
            ),
            false => (
                [lhs_shape[lhs_dims - 2], lhs_shape[lhs_dims - 1]],
                [lhs_stride[lhs_dims - 2], lhs_stride[lhs_dims - 1]],
            ),
        };

        let rhs_dims = rhs_shape.len();
        let (rhs_shape, rhs_stride) = match trans_b {
            true => (
                [rhs_shape[rhs_dims - 1], rhs_shape[rhs_dims - 2]],
                [rhs_stride[rhs_dims - 1], rhs_stride[rhs_dims - 2]],
            ),
            false => (
                [rhs_shape[rhs_dims - 2], rhs_shape[rhs_dims - 1]],
                [rhs_stride[rhs_dims - 2], rhs_stride[rhs_dims - 1]],
            ),
        };

        trace!(
            "lhs_shape={lhs_shape:?},\
             lhs_stride={lhs_stride:?},\
             rhs_shape={rhs_shape:?},\
             rhs_stride={rhs_stride:?}\
             m={m:?},\
             k={k:?},\
             n={n:?}",
        );

        Self {
            lhs_shape,
            lhs_stride,
            rhs_shape,
            rhs_stride,
            b,
            m,
            n,
            k,
        }
    }

    pub fn calculate_output_shape(&self) -> [usize; 3] {
        [self.b, self.m, self.n]
    }

    pub fn strided_batch_config<T>(
        &self,
        (alpha, beta): (T, T),
    ) -> Result<StridedBatchedConfig<T>> {
        gemm_config::<T>(
            alpha,
            beta,
            (self.b, self.m, self.n, self.k),
            (&self.lhs_shape, &self.lhs_stride),
            (&self.rhs_shape, &self.rhs_stride),
        )
    }

    pub fn compute_f32(
        &self,
        device: Arc<CudaDevice>,
        lhs_data: &CudaSlice<f32>,
        rhs_data: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        config: StridedBatchedConfig<f32>,
    ) -> Result<()> {
        let cublas = CudaBlas::new(device)?;

        unsafe {
            gemm_stride_batched_f32(
                &cublas,
                config,
                &rhs_data.slice(..),
                &lhs_data.slice(..),
                out,
            )?;
        };

        Ok(())
    }
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
        _ => return Err(Error::InvalidInputShapes),
    };

    let lhs_stride = lhs_layout.1;
    let (transb, ldb) = match lhs_stride {
        [.., m_stride, 1] | [m_stride, 1] if *m_stride == k => {
            (sys::cublasOperation_t::CUBLAS_OP_N, k)
        }
        [.., 1, m_stride] | [1, m_stride] if *m_stride == k => {
            (sys::cublasOperation_t::CUBLAS_OP_T, m)
        }
        // Todo: return an non-contiguous error.
        _ => return Err(Error::InvalidInputShapes),
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
    .map_err(Into::into)
}

pub unsafe fn _gemm_stride_batched_f16(
    cublas: &CudaBlas,
    config: StridedBatchedConfig<f16>,
    a: &CudaView<f16>,
    b: &CudaView<f16>,
    c: &mut CudaSlice<f16>,
) -> Result<()> {
    let alpha = &config.gemm.alpha as *const f16 as *const _;
    let beta = &config.gemm.beta as *const f16 as *const _;

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        config.gemm.transa,
        config.gemm.transb,
        config.gemm.m,
        config.gemm.n,
        config.gemm.k,
        alpha,
        *a.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_16F,
        config.gemm.lda,
        config.stride_a,
        *b.device_ptr() as *const _,
        sys::cudaDataType_t::CUDA_R_16F,
        config.gemm.ldb,
        config.stride_b,
        beta,
        *c.device_ptr_mut() as *mut _,
        sys::cudaDataType_t::CUDA_R_16F,
        config.gemm.ldc,
        config.stride_c,
        config.batch_size,
        sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    .map_err(Into::into)
}

#[cfg(test)]
mod test {
    use crate::kernels::gemm::GemmOp;
    use crate::utils;
    use cudarc::driver::CudaDevice;

    #[test]
    fn test_gemm_f32() {
        let device = CudaDevice::new(0).unwrap();

        let lhs_shape = vec![1, 2, 2];
        let mut lhs_stride = vec![0; lhs_shape.len()];
        utils::calculate_stride(&lhs_shape, &mut lhs_stride);
        let lhs_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let rhs_shape = vec![1, 2, 2];
        let mut rhs_stride = vec![0; rhs_shape.len()];
        utils::calculate_stride(&rhs_shape, &mut rhs_stride);
        let rhs_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let op = GemmOp::new(
            &lhs_shape,
            &lhs_stride,
            &rhs_shape,
            &rhs_stride,
            false,
            false,
        );
        let config = op.strided_batch_config((1.0, 0.0)).unwrap();

        let output_size = op.calculate_output_shape().iter().product();
        let mut out = device.alloc_zeros(output_size).unwrap();

        op.compute_f32(device.clone(), &lhs_data, &rhs_data, &mut out, config)
            .unwrap();

        let result = device.dtoh_sync_copy(&out).unwrap();

        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    }
}

/*

       x = [
           [ a, b, c],
           [ d, e, f],
           ];




   shape = [ 2, 3 ]
   stride = [ 3, 1 ]
   mem = [a, b, c, d, e, f]

   0*3 + 2*1 = 2
   1*3 + 1*1 = 4


   x' = [
           [ a, d],
           [ b, e],
           [ c, f],
       ]

      shape = [3, 2]
      stride = [1, 3]
      mem = [a, b, c, d, e, f]

       0*1 + 1*3 = 3
       1*1 + 0*3 = 1
       1*1 + 1*3 = 4
*/
