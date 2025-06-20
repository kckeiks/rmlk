use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
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
    fn prepare_gemm_params(&self, ctx: &mut Context<Cuda>) -> Result<MatMulParams> {
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

    fn compute_output_shape(&self, params: &MatMulParams, ctx: &mut Context<Cuda>) -> Result<()> {
        let y = ctx.get_output(0)?;
        let y_index = y.dst_id();

        let add_batch = {
            let a = ctx.get_input(0)?;
            let b = ctx.get_input(1)?;

            if a.shape().len() >= 3 || b.shape().len() >= 3 {
                true
            } else {
                false
            }
        };

        match params.promoted {
            None => {
                if add_batch {
                    ctx.execution_state_mut().copy_shape_from_slice(
                        &[params.gemm.b, params.gemm.m, params.gemm.n],
                        y_index,
                    )?;
                } else {
                    ctx.execution_state_mut()
                        .copy_shape_from_slice(&[params.gemm.m, params.gemm.n], y_index)?;
                }
            }
            Some(Promoted::Both) => {
                ctx.execution_state_mut()
                    .copy_shape_from_slice(&[], y_index)?;
            }
            Some(Promoted::Left) => {
                if add_batch {
                    ctx.execution_state_mut()
                        .copy_shape_from_slice(&[params.gemm.b, params.gemm.n], y_index)?;
                } else {
                    ctx.execution_state_mut()
                        .copy_shape_from_slice(&[params.gemm.b, params.gemm.n], y_index)?;
                }
            }
            Some(Promoted::Right) => {
                if add_batch {
                    ctx.execution_state_mut()
                        .copy_shape_from_slice(&[params.gemm.b, params.gemm.m], y_index)?;
                } else {
                    ctx.execution_state_mut()
                        .copy_shape_from_slice(&[params.gemm.b, params.gemm.m], y_index)?;
                }
            }
        }

        Ok(())
    }

    fn compute_multiplication<D>(
        &self,
        params: &MatMulParams,
        ctx: &mut Context<Cuda>,
    ) -> Result<()>
    where
        D: CudaParamMap
            + DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + TryFrom<f32>,
    {
        self.compute_output_shape(&params, ctx)?;
        let config = gemm::strided_batch_config::<D>((D::one(), D::zero()), &params.gemm)?;

        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;
        let y = ctx.get_output(0)?;

        debug!("[a][shape={:?}][stride=[{:?}]", a.shape(), a.stride());
        debug!("[b][shape={:?}][stride=[{:?}]", b.shape(), b.stride());
        debug!("[y][shape={:?}][stride=[{:?}]", y.shape(), y.stride());

        let a_expected_size = params.gemm.b * params.gemm.matrix_a_shape.iter().product::<usize>();
        let a_size = a.shape().iter().product::<usize>();
        let a_broadcast_slice = if params.gemm.b > 1 && a_expected_size != a_size {
            let mut slice = self
                .stream
                .alloc_zeros::<D>(params.gemm.b * a_size)
                .map_err(|e| InternalError::Device { error: e.into() })?;
            for batch_i in 0..params.gemm.b {
                let a_dev_data_ref = a.try_dev_data_ptr()?;
                let a_dev_data = a_dev_data_ref.data::<D>();
                debug_assert_eq!(a_size, a_dev_data.len());
                self.stream
                    .memcpy_dtod(
                        a_dev_data.as_ref(),
                        &mut slice.slice_mut(batch_i * a_size..batch_i * a_size + a_size),
                    )
                    .map_err(|e| InternalError::Device { error: e.into() })?;
            }

            Some(slice)
        } else {
            None
        };

        let b_expected_size = params.gemm.b * params.gemm.matrix_b_shape.iter().product::<usize>();
        let b_size = b.shape().iter().product::<usize>();
        let b_broadcast_slice = if params.gemm.b > 1 && b_expected_size != b_size {
            let mut slice = self
                .stream
                .alloc_zeros::<D>(params.gemm.b * b_size)
                .map_err(|e| InternalError::Device { error: e.into() })?;
            for batch_i in 0..params.gemm.b {
                let b_dev_data_ref = b.try_dev_data_ptr()?;
                let b_dev_data = b_dev_data_ref.data::<D>();
                debug_assert_eq!(b_size, b_dev_data.len());
                self.stream
                    .memcpy_dtod(
                        b_dev_data.as_ref(),
                        &mut slice.slice_mut(batch_i * b_size..batch_i * b_size + b_size),
                    )
                    .map_err(|e| InternalError::Device { error: e.into() })?;
            }

            Some(slice)
        } else {
            None
        };

        let output_tensor = ctx.get_output(0)?;
        common::init_tensor_device_data::<D>(&self.stream, output_tensor)?;

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        match (a_broadcast_slice, b_broadcast_slice) {
            (Some(a_broadcast_slice), Some(b_broadcast_slice)) => {
                gemm::compute::<D>(
                    &self.stream,
                    &a_broadcast_slice,
                    &b_broadcast_slice,
                    &mut y_dev_data,
                    config,
                )?;
            }
            (Some(a_broadcast_slice), None) => {
                let b_dev_data_ref = b.try_dev_data_ptr()?;
                let b_dev_data = b_dev_data_ref.data::<D>();

                gemm::compute::<D>(
                    &self.stream,
                    &a_broadcast_slice,
                    &b_dev_data,
                    &mut y_dev_data,
                    config,
                )?;
            }
            (None, Some(b_broadcast_slice)) => {
                let a_dev_data_ref = a.try_dev_data_ptr()?;
                let a_dev_data = a_dev_data_ref.data::<D>();

                gemm::compute::<D>(
                    &self.stream,
                    &a_dev_data,
                    &b_broadcast_slice,
                    &mut y_dev_data,
                    config,
                )?;
            }
            (None, None) => {
                let a_dev_data_ref = a.try_dev_data_ptr()?;
                let a_dev_data = a_dev_data_ref.data::<D>();
                let b_dev_data_ref = b.try_dev_data_ptr()?;
                let b_dev_data = b_dev_data_ref.data::<D>();

                gemm::compute::<D>(
                    &self.stream,
                    &a_dev_data,
                    &b_dev_data,
                    &mut y_dev_data,
                    config,
                )?;
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
            + TryFrom<f32>,
    {
        let params = self.prepare_gemm_params(ctx)?;
        self.compute_multiplication::<D>(&params, ctx)?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            // Todo: add support.
            // DataType::Float16 => self.compute_matmul::<f16>(ctx),
            DataType::Float => self.compute_matmul::<f32>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
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
