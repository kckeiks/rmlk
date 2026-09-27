use crate::core::error::UnsupportedDataType;
use crate::core::Context;

use crate::providers::cuda::data::CudaData;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::{Cuda, Tensor};
use crate::utils::FromF32;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::gemm;
use rmlk_cuda::kernels::gemm::GemmParams;
use rmlk_cuda::params::CudaParamMap;
use rmlk_schema::{DataType, DataTypeMap};
use std::cmp;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[derive(Debug)]
enum Promoted {
    Left,
    Right,
    Both,
}

struct MatMulParams {
    gemm: GemmParams,
    promoted: Option<Promoted>,
}

pub struct MatMulBackend {
    stream: Arc<CudaStream>,
}

impl MatMulBackend {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            stream: stream.clone(),
        }
    }
}

impl MatMulBackend {
    fn broadcast_shape<T>(
        &self,
        ctx: &Context<Cuda>,
        tensor: &Tensor,
        size: usize,
        target_size: usize,
        params: &MatMulParams,
    ) -> Result<Option<CudaData>>
    where
        T: CudaParamMap
            + DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + FromF32,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let data = if params.gemm.b > 1 && target_size != size {
            let mut on_dev_buf = cuda_bump.alloc_with_fallback::<T>(params.gemm.b * size)?;
            {
                let mut slice = on_dev_buf.data_mut();
                for batch_i in 0..params.gemm.b {
                    let payload = tensor.payload();
                    let data = payload.data::<T>();
                    assert_eq!(size, data.len());
                    self.stream.memcpy_dtod(
                        data.as_ref(),
                        &mut slice.slice_mut(batch_i * size..batch_i * size + size),
                    )?;
                }
            }
            Some(on_dev_buf)
        } else {
            None
        };

        Ok(data)
    }

    fn compute_multiplication<T>(
        &self,
        params: &MatMulParams,
        ctx: &mut Context<Cuda>,
    ) -> Result<()>
    where
        T: CudaParamMap
            + DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + FromF32,
    {
        compute_output_shape(params, ctx)?;
        let config = gemm::strided_batch_config::<T>((T::one(), T::zero()), &params.gemm)?;

        let a = ctx.get_input(0)?;

        debug!(
            "[a][dtype={:?}][shape={:?}][stride=[{:?}]",
            a.dtype(),
            a.shape(),
            a.stride()
        );

        let b = ctx.get_input(1)?;

        debug!(
            "[b][dtype={:?}][shape={:?}][stride=[{:?}]",
            b.dtype(),
            b.shape(),
            b.stride()
        );

        let y = ctx.get_output(0)?;

        if a.rank() == 1 && b.rank() == 1 {
            y.init_scalar_payload::<T>()?;
        } else {
            y.init_payload::<T>()?;
        }

        debug!(
            "[y][dtype={:?}][shape={:?}][stride=[{:?}]",
            y.dtype(),
            y.shape(),
            y.stride()
        );

        let a_expected_size = params.gemm.b * params.gemm.matrix_a_shape.iter().product::<usize>();
        let a_size = a.shape().iter().product::<usize>();
        let a_broadcast_slice =
            self.broadcast_shape::<T>(ctx, &a, a_size, a_expected_size, params)?;

        let b_expected_size = params.gemm.b * params.gemm.matrix_b_shape.iter().product::<usize>();
        let b_size = b.shape().iter().product::<usize>();
        let b_broadcast_slice =
            self.broadcast_shape::<T>(ctx, &b, b_size, b_expected_size, params)?;

        let y = ctx.get_output(0)?;
        let mut y_payload = y.payload_mut();
        let mut y_data = y_payload.data_mut();

        match (a_broadcast_slice, b_broadcast_slice) {
            (Some(a_broadcast_slice), Some(b_broadcast_slice)) => {
                gemm::compute::<T>(
                    &self.stream,
                    &a_broadcast_slice.data::<T>(),
                    &b_broadcast_slice.data::<T>(),
                    &mut y_data,
                    config,
                )?;
            }
            (Some(a_broadcast_slice), None) => {
                let b_payload = b.payload();
                let b_data = b_payload.data::<T>();

                gemm::compute::<T>(
                    &self.stream,
                    &a_broadcast_slice.data::<T>(),
                    &b_data,
                    &mut y_data,
                    config,
                )?;
            }
            (None, Some(b_broadcast_slice)) => {
                let a_payload = a.payload();
                let a_data = a_payload.data::<T>();

                gemm::compute::<T>(
                    &self.stream,
                    &a_data,
                    &b_broadcast_slice.data::<T>(),
                    &mut y_data,
                    config,
                )?;
            }
            (None, None) => {
                let a_payload = a.payload();
                let a_data = a_payload.data::<T>();
                let b_payload = b.payload();
                let b_data = b_payload.data::<T>();

                gemm::compute::<T>(&self.stream, &a_data, &b_data, &mut y_data, config)?;
            }
        }

        Ok(())
    }

    pub fn compute_matmul<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: CudaParamMap
            + DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + FromF32,
    {
        let params = prepare_gemm_params(ctx)?;
        self.compute_multiplication::<D>(&params, ctx)?;

        #[cfg(feature = "dump")]
        debug::write_results_binary::<D, D, D>(
            "debugging/matmul",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_matmul::<f16>(ctx),
            DataType::Float => self.compute_matmul::<f32>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

fn prepare_gemm_params(ctx: &mut Context<Cuda>) -> Result<MatMulParams> {
    let a = ctx.get_input(0)?;
    let b = ctx.get_input(1)?;
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_rank = a.shape().len();
    let b_rank = b.shape().len();

    if a_rank == 0 || b_rank == 0 {
        return Err(MatMulError::ZeroRank.into());
    }

    let (promoted_a, batch_a, a_2d_shape, a_2d_stride) = if a_rank == 1 {
        (true, 1, [1, a_shape[0]], [a_shape[0], 1])
    } else if a_rank == 2 {
        (false, 1, [a_shape[0], a_shape[1]], [a_shape[1], 1])
    } else {
        let batch = a_shape[..a_rank - 2].iter().product::<usize>();
        (
            false,
            batch,
            [a_shape[a_rank - 2], a_shape[a_rank - 1]],
            [a_shape[a_rank - 1], 1],
        )
    };

    let (promoted_b, batch_b, b_2d_shape, b_2d_stride) = if b_rank == 1 {
        (true, 1, [b_shape[0], 1], [1, 1])
    } else if b_rank == 2 {
        (false, 1, [b_shape[0], b_shape[1]], [b_shape[1], 1])
    } else {
        let batch = b_shape[..b_rank - 2].iter().product::<usize>();
        (
            false,
            batch,
            [b_shape[b_rank - 2], b_shape[b_rank - 1]],
            [b_shape[b_rank - 1], 1],
        )
    };

    let promoted = match (promoted_a, promoted_b) {
        (true, true) => Some(Promoted::Both),
        (false, true) => Some(Promoted::Right),
        (true, false) => Some(Promoted::Left),
        (false, false) => None,
    };

    if a_2d_shape[1] != b_2d_shape[0] {
        // Todo: we need a better error.
        return Err(MatMulError::IncompatibleDimForMul.into());
    }

    if batch_a != batch_b && batch_a != 1 && batch_b != 1 {
        // Todo: we need a better error.
        return Err(MatMulError::IncompatibleDimForBroadcast.into());
    }

    let gemm_params = gemm::gemm_params(
        &a_2d_shape,
        &a_2d_stride,
        &b_2d_shape,
        &b_2d_stride,
        false,
        false,
        cmp::max(batch_a, batch_b),
    )?;

    Ok(MatMulParams {
        gemm: gemm_params,
        promoted,
    })
}

fn compute_output_shape(params: &MatMulParams, ctx: &Context<Cuda>) -> Result<()> {
    let y = ctx.get_output(0)?;

    let (add_batch, max_rank) = {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        let add_batch = a.shape().len() >= 3 || b.shape().len() >= 3;

        // We do this to appease the compiler.
        let res = (add_batch, cmp::max(a.shape().len(), b.shape().len()));
        res
    };

    match params.promoted {
        None => {
            if add_batch {
                let scratch_alloc = ctx.execution_state().scratch_alloc().clone();
                let output_shape = scratch_alloc.allocate_fill(max_rank, 1)?;
                output_shape[max_rank - 3] = params.gemm.b;
                output_shape[max_rank - 2] = params.gemm.m;
                output_shape[max_rank - 1] = params.gemm.n;

                y.copy_shape_from_slice(output_shape);
            } else {
                y.copy_shape_from_slice(&[params.gemm.m, params.gemm.n]);
            }
        }
        Some(Promoted::Both) => {
            // Todo: handle scalars.
            y.copy_shape_from_slice(&[]);
        }
        Some(Promoted::Left) => {
            if add_batch {
                y.copy_shape_from_slice(&[params.gemm.b, params.gemm.n]);
            } else {
                y.copy_shape_from_slice(&[params.gemm.n]);
            }
        }
        Some(Promoted::Right) => {
            if add_batch {
                y.copy_shape_from_slice(&[params.gemm.b, params.gemm.m]);
            } else {
                y.copy_shape_from_slice(&[params.gemm.m]);
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum MatMulError {
    ZeroRank,
    IncompatibleDimForMul,
    IncompatibleDimForBroadcast,
}

impl Display for MatMulError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for MatMulError {}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::Op;

    #[test]
    fn d1_d1() {
        let out = OpTest::new(Op::MatMul)
            .input([4], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([4], vec![5.0f32, 6.0, 7.0, 8.0])
            .output([])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![70.0]);
    }

    #[test]
    fn d1_d2() {
        let out = OpTest::new(Op::MatMul)
            .input([3], vec![1.0f32, 2.0, 3.0])
            .input(
                [3, 4],
                vec![
                    4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                ],
            )
            .output([4])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![56.0, 62.0, 68.0, 74.0]);
    }

    #[test]
    fn d1_d3() {
        let out = OpTest::new(Op::MatMul)
            .input([3], vec![1.0f32, 2.0, 3.0])
            .input(
                [2, 3, 4],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 2.0, 4.0,
                    6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 0.0, -1.0, -2.0, -3.0,
                ],
            )
            .output([2, 4])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![38.0, 44.0, 50.0, 56.0, 4.0, 7.0, 10.0, 13.0]);
    }

    #[test]
    fn d2_d1() {
        let out = OpTest::new(Op::MatMul)
            .input([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .input([3], vec![7.0f32, 8.0, 9.0])
            .output([2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![50.0, 122.0]);
    }

    #[test]
    fn d2_d2() {
        let out = OpTest::new(Op::MatMul)
            .input([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .input(
                [3, 4],
                vec![
                    7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                ],
            )
            .output([2, 4])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
        );
    }

    #[test]
    fn d2_d3() {
        let out = OpTest::new(Op::MatMul)
            .input([2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .input(
                [2, 3, 2],
                vec![
                    7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                ],
            )
            .output([2, 2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![58.0, 64.0, 139.0, 154.0, 94.0, 100.0, 229.0, 244.0]
        );
    }

    #[test]
    fn d3_d1() {
        let out = OpTest::new(Op::MatMul)
            .input(
                [2, 3, 3],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
                    3.0, 2.0, 1.0,
                ],
            )
            .input([3], vec![1.0f32, 2.0, 3.0])
            .output([2, 3])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![14.0, 32.0, 50.0, 46.0, 28.0, 10.0]);
    }

    #[test]
    fn d3_d2() {
        let out = OpTest::new(Op::MatMul)
            .input(
                [2, 2, 3],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            )
            .input([3, 2], vec![13.0f32, 14.0, 15.0, 16.0, 17.0, 18.0])
            .output([2, 2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![94.0, 100.0, 229.0, 244.0, 364.0, 388.0, 499.0, 532.0]
        );
    }

    #[test]
    fn d3_d3() {
        let out = OpTest::new(Op::MatMul)
            .input(
                [2, 2, 3],
                vec![
                    1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            )
            .input(
                [2, 3, 2],
                vec![
                    13.0f32, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                ],
            )
            .output([2, 2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![94.0, 100.0, 229.0, 244.0, 508.0, 532.0, 697.0, 730.0]
        );
    }

    #[test]
    fn d4_shape_and_values() {
        // Former matmul_4d only asserted output shape via Shape; check values too.
        let out = OpTest::new(Op::MatMul)
            .input([1, 24, 6, 6], vec![1.0f32; 24 * 6 * 6])
            .input([1, 24, 6, 128], vec![13.0f32; 24 * 6 * 128])
            .output([1, 24, 6, 128])
            .run::<f32>()
            .unwrap();
        assert_eq!(out.len(), 24 * 6 * 128);
        assert!(out.iter().all(|&x| x == 78.0));
    }

    #[test]
    fn broadcast_batch() {
        let out = OpTest::new(Op::MatMul)
            .input([1, 2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .input(
                [2, 3, 2],
                vec![
                    7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                ],
            )
            .output([2, 2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![58.0, 64.0, 139.0, 154.0, 94.0, 100.0, 229.0, 244.0]
        );
    }

    #[test]
    fn batched_2d_vs_reference() {
        // Former matmul_large (was commented out); small [1,4,8] x [8,6].
        const N_MID: usize = 4;
        const K: usize = 8;
        const N: usize = 6;
        let a: Vec<f32> = (0..N_MID * K).map(|i| i as f32).collect();
        // Integer-valued B so GPU and host f32 agree exactly.
        let b: Vec<f32> = (0..K)
            .flat_map(|r| (0..N).map(move |c| (r + c) as f32))
            .collect();
        let mut expected = vec![0f32; N_MID * N];
        for i in 0..N_MID {
            for k in 0..K {
                let a_val = a[i * K + k];
                for j in 0..N {
                    expected[i * N + j] += a_val * b[k * N + j];
                }
            }
        }
        let out = OpTest::new(Op::MatMul)
            .input([1, N_MID, K], a)
            .input([K, N], b)
            .output([1, N_MID, N])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn rejects_i32() {
        let err = OpTest::new(Op::MatMul)
            .input([2, 2], vec![1i32, 2, 3, 4])
            .input([2, 2], vec![1i32, 0, 0, 1])
            .output([2, 2])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
