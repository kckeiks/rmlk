use crate::error::Error;
use crate::error::Result;
use crate::params::CudaParamMap;
use cudarc::cublas::{sys, CudaBlas, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DevicePtr, DevicePtrMut};
use log::trace;
use std::sync::Arc;

pub struct GemmParams {
    pub matrix_a_shape: [usize; 2],
    pub matrix_a_stride: [usize; 2],
    pub matrix_b_shape: [usize; 2],
    pub matrix_b_stride: [usize; 2],
    pub matrix_output_shape: [usize; 2],
    pub b: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub fn gemm_params(
    a_shape: &[usize],
    a_stride: &[usize],
    b_shape: &[usize],
    b_stride: &[usize],
    trans_a: bool,
    trans_b: bool,
    b: usize,
) -> Result<GemmParams> {
    let (m, k) = match trans_a {
        true => {
            let m = a_shape[a_shape.len() - 1];
            let k = a_shape[a_shape.len() - 2];
            (m, k)
        }
        false => {
            let m = a_shape[a_shape.len() - 2];
            let k = a_shape[a_shape.len() - 1];
            (m, k)
        }
    };

    let n = match trans_b {
        false => b_shape[b_shape.len() - 1],
        true => b_shape[b_shape.len() - 2],
    };

    let a_dims = a_shape.len();
    let (matrix_a_shape, matrix_a_stride) = match trans_a {
        true => (
            // We perform a logical transpose.
            [a_shape[a_dims - 1], a_shape[a_dims - 2]],
            [a_stride[a_dims - 1], a_stride[a_dims - 2]],
        ),
        false => (
            [a_shape[a_dims - 2], a_shape[a_dims - 1]],
            [a_stride[a_dims - 2], a_stride[a_dims - 1]],
        ),
    };

    let b_dims = b_shape.len();
    let (matrix_b_shape, matrix_b_stride) = match trans_b {
        true => (
            // We perform a logical transpose.
            [b_shape[b_dims - 1], b_shape[b_dims - 2]],
            [b_stride[b_dims - 1], b_stride[b_dims - 2]],
        ),
        false => (
            [b_shape[b_dims - 2], b_shape[b_dims - 1]],
            [b_stride[b_dims - 2], b_stride[b_dims - 1]],
        ),
    };

    trace!(
        "matrix_a_shape={matrix_a_shape:?},\
             matrix_a_stride={matrix_a_stride:?},\
             matrix_b_shape={matrix_b_shape:?},\
             matrix_b_stride={matrix_b_stride:?}\
             m={m:?},\
             k={k:?},\
             n={n:?}",
    );

    Ok(GemmParams {
        matrix_a_shape,
        matrix_a_stride,
        matrix_b_shape,
        matrix_b_stride,
        matrix_output_shape: [m, n],
        b,
        m,
        n,
        k,
    })
}

pub fn strided_batch_config<T>(
    (alpha, beta): (T, T),
    gemm_params: &GemmParams,
) -> Result<StridedBatchedConfig<T>> {
    gemm_config::<T>(
        alpha,
        beta,
        (gemm_params.b, gemm_params.m, gemm_params.n, gemm_params.k),
        &gemm_params.matrix_a_shape,
        &gemm_params.matrix_a_stride,
        &gemm_params.matrix_b_shape,
        &gemm_params.matrix_b_stride,
    )
}

pub fn compute<T: CudaParamMap>(
    device: Arc<CudaDevice>,
    a_data: &CudaSlice<T>,
    b_data: &CudaSlice<T>,
    y_data: &mut CudaSlice<T>,
    config: StridedBatchedConfig<T>,
) -> Result<()> {
    let cublas = CudaBlas::new(device)?;

    // NOTE: We pass b_data as the `A` pointer for cuBLAS, and a_data as the `B` pointer.
    // This is because the shape/stride logic in gemm_config is reversed for
    // handling cuBLAS's column-major assumption.
    unsafe {
        gemm_stride_batched::<T>(
            &cublas,
            config,
            &b_data.slice(..),
            &a_data.slice(..),
            y_data,
        )?;
    };

    Ok(())
}

fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    a_shape: &[usize],
    a_stride: &[usize],
    b_shape: &[usize],
    b_stride: &[usize],
) -> Result<StridedBatchedConfig<T>> {
    let (transa, lda) = match infer_matrix_layout(b_shape, b_stride)? {
        MatrixLayout::RowMajor { cols, .. } => {
            debug_assert_eq!(cols, n);
            (sys::cublasOperation_t::CUBLAS_OP_N, cols)
        }
        MatrixLayout::ColumnMajor { rows, .. } => {
            debug_assert_eq!(rows, k);
            (sys::cublasOperation_t::CUBLAS_OP_T, rows)
        }
    };

    let (transb, ldb) = match infer_matrix_layout(a_shape, a_stride)? {
        MatrixLayout::RowMajor { cols, .. } => {
            debug_assert_eq!(cols, k);
            (sys::cublasOperation_t::CUBLAS_OP_N, cols)
        }
        MatrixLayout::ColumnMajor { rows, .. } => {
            debug_assert_eq!(rows, m);
            (sys::cublasOperation_t::CUBLAS_OP_T, rows)
        }
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

pub unsafe fn gemm_stride_batched<T: CudaParamMap>(
    cublas: &CudaBlas,
    config: StridedBatchedConfig<T>,
    a: &CudaView<T>,
    b: &CudaView<T>,
    y: &mut CudaSlice<T>,
) -> Result<()> {
    let alpha = &config.gemm.alpha as *const T as *const _;
    let beta = &config.gemm.beta as *const T as *const _;

    cudarc::cublas::result::gemm_strided_batched_ex(
        *cublas.handle(),
        config.gemm.transa,
        config.gemm.transb,
        config.gemm.m,
        config.gemm.n,
        config.gemm.k,
        alpha,
        *a.device_ptr() as *const _,
        T::data_type(),
        config.gemm.lda,
        config.stride_a,
        *b.device_ptr() as *const _,
        T::data_type(),
        config.gemm.ldb,
        config.stride_b,
        beta,
        *y.device_ptr_mut() as *mut _,
        T::data_type(),
        config.gemm.ldc,
        config.stride_c,
        config.batch_size,
        T::cublas_compute_type(),
        T::cublas_gemma_algo(),
    )
    .map_err(Into::into)
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

pub enum MatrixLayout {
    RowMajor { cols: usize, rows: usize },
    ColumnMajor { cols: usize, rows: usize },
}

fn infer_matrix_layout(shape: &[usize], stride: &[usize]) -> Result<MatrixLayout> {
    assert_eq!(shape.len(), stride.len());
    assert_eq!(shape.len(), 2);

    let (rows, cols) = (shape[0], shape[1]);

    match stride {
        [stride_row, 1] if *stride_row == cols => Ok(MatrixLayout::RowMajor { rows, cols }),
        [1, stride_col] if *stride_col == rows => Ok(MatrixLayout::ColumnMajor { rows, cols }),
        _ => Err(Error::NonContiguousMemory(format!(
            "Strides {stride:?} do not match row-major or column-major for shape {shape:?}"
        ))),
    }
}

#[cfg(test)]
mod test {
    use crate::kernels::gemm::{compute, gemm_params, strided_batch_config};
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

        let output_shape = vec![1, 2, 2];

        let params = gemm_params(
            &lhs_shape,
            &lhs_stride,
            &rhs_shape,
            &rhs_stride,
            false,
            false,
            1,
        )
        .unwrap();
        let config = strided_batch_config((1.0, 0.0), &params).unwrap();

        let output_size = output_shape.iter().product();
        let mut out = device.alloc_zeros(output_size).unwrap();

        compute::<f32>(device.clone(), &lhs_data, &rhs_data, &mut out, config).unwrap();

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
