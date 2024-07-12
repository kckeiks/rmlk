use crate::cuda::data::CudaData;
use crate::kernel::Context;
use crate::Error;
use crate::Result;
use cudarc::cublas::{sys, CudaBlas, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaDevice, CudaSlice, CudaView, DevicePtr, DevicePtrMut};
use half::f16;
use rmlk_ir::{Attribute, DataType};
use std::collections::HashMap;
use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let lhs = ctx.get_input(0)?;
    let rhs = ctx.get_input(1)?;

    let attr = GemmAttributes::new(ctx.get_attributes().ok_or(Error::MissingAttributes)?)?;

    let lhs_shape = lhs.shape();
    let b = lhs_shape[..lhs_shape.len() - 2].iter().product::<usize>();
    let (m, k) = match attr.trans_a {
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

    let rhs_shape = rhs.shape();
    let n = match attr.trans_b {
        true => rhs_shape[rhs_shape.len() - 1],
        false => rhs_shape[rhs_shape.len() - 2],
    };

    let lhs_stride = lhs.stride();
    let lhs_dims = lhs_shape.len();
    let (lhs_shape, lhs_stride) = match attr.trans_a {
        true => (
            [lhs_shape[lhs_dims - 1], lhs_shape[lhs_dims - 2]],
            [lhs_stride[lhs_dims - 1], lhs_stride[lhs_dims - 2]],
        ),
        false => (
            [lhs_shape[lhs_dims - 2], lhs_shape[lhs_dims - 1]],
            [lhs_stride[lhs_dims - 2], lhs_stride[lhs_dims - 1]],
        ),
    };

    let rhs_stride = rhs.stride();
    let rhs_dims = rhs_shape.len();
    let (rhs_shape, rhs_stride) = match attr.trans_b {
        true => (
            [rhs_shape[rhs_dims - 1], rhs_shape[rhs_dims - 2]],
            [rhs_stride[rhs_dims - 1], rhs_stride[rhs_dims - 2]],
        ),
        false => (
            [rhs_shape[rhs_dims - 2], rhs_shape[rhs_dims - 1]],
            [rhs_stride[rhs_dims - 2], rhs_stride[rhs_dims - 1]],
        ),
    };

    match *lhs.dtype() {
        DataType::Float16 => {
            todo!()
        }
        DataType::Float => {
            let config = gemm_config::<f32>(
                attr.alpha,
                attr.beta,
                (b, m, n, k),
                (&lhs_shape, &lhs_stride),
                (&rhs_shape, &rhs_stride),
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
    .map_err(|_| Error::Unknown)
}

pub struct GemmAttributes {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

impl GemmAttributes {
    pub fn new(attrs: &HashMap<Box<str>, Attribute>) -> Result<Self> {
        let mut alpha = None;
        let mut beta = None;
        let mut trans_a = None;
        let mut trans_b = None;

        if let Some(attr) = attrs.get("alpha") {
            alpha = attr.float();
        }

        if let Some(attr) = attrs.get("beta") {
            beta = attr.float();
        }

        if let Some(attr) = attrs.get("transA") {
            trans_a = match attr.int() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                None => None,
                _ => return Err(Error::InvalidAttribute),
            };
        }

        if let Some(attr) = attrs.get("transB") {
            trans_b = match attr.int() {
                Some(0) => Some(false),
                Some(1) => Some(true),
                None => None,
                _ => return Err(Error::InvalidAttribute),
            };
        }

        Ok(Self {
            alpha: alpha.unwrap_or(1.0),
            beta: beta.unwrap_or(1.0),
            trans_a: trans_a.unwrap_or(false),
            trans_b: trans_b.unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod test {
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::CudaKernel;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    // use half::f16;
    use rmlk_ir::{DataType, Op};

    // #[test]
    // fn test_gemm_f16() {
    //     let device = CudaDevice::new(0).unwrap();
    //
    //     let shape = vec![1, 2, 2];
    //     let dtype = DataType::Float16;
    //     let op = Op::Gemm;
    //
    //     let node_a = TestNode {
    //         shape: shape.clone(),
    //         dtype,
    //         data: Some(CudaData::F16(
    //             device
    //                 .htod_copy(vec![
    //                     f16::from_f32(1.0),
    //                     f16::from_f32(2.0),
    //                     f16::from_f32(3.0),
    //                     f16::from_f32(4.0),
    //                 ])
    //                 .unwrap(),
    //         )),
    //     };
    //     let node_b = TestNode {
    //         shape: shape.clone(),
    //         dtype,
    //         data: Some(CudaData::F16(
    //             device
    //                 .htod_copy(vec![
    //                     f16::from_f32(1.0),
    //                     f16::from_f32(2.0),
    //                     f16::from_f32(3.0),
    //                     f16::from_f32(4.0),
    //                 ])
    //                 .unwrap(),
    //         )),
    //     };
    //
    //     let node_c = TestNode {
    //         shape,
    //         dtype,
    //         data: None,
    //     };
    //
    //     let params = TestParams {
    //         inputs: vec![node_a, node_b],
    //         outputs: vec![node_c],
    //         attributes: vec![],
    //         op,
    //     };
    //
    //     let (_, state) = test_utils::build_graph_and_state(params);
    //     let mut context = Context::new(state, 2).unwrap();
    //
    //     let cuda_kernel = CudaKernel::new(op, device.clone());
    //     cuda_kernel.compute(&mut context).unwrap();
    //
    //     let out_data = context
    //         .get_output(0)
    //         .unwrap()
    //         .data()
    //         .unwrap()
    //         .f16()
    //         .unwrap();
    //     let result = device.dtoh_sync_copy(out_data).unwrap();
    //
    //     assert_eq!(
    //         result,
    //         vec![
    //             f16::from_f32(7.0),
    //             f16::from_f32(10.0),
    //             f16::from_f32(15.0),
    //             f16::from_f32(22.0)
    //         ]
    //     )
    // }

    #[test]
    fn test_gemm_f32() {
        let device = CudaDevice::new(0).unwrap();

        let shape = vec![1, 2, 2];
        let dtype = DataType::Float;
        let op = Op::Gemm;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes: vec![],
            op,
        };

        let (_, state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(op, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    }
}
