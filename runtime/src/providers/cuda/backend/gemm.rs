use crate::attributes::gemm::GemmAttributes;
use crate::core::error::InternalError;
use crate::core::Context;

use crate::providers::cuda::Cuda;
use crate::utils::FromF32;
use anyhow::Result;
use cudarc::cublas::StridedBatchedConfig;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::add::AddKernel;
use rmlk_cuda::kernels::gemm::GemmParams;
use rmlk_cuda::kernels::{add, binary, gemm};
use rmlk_cuda::params::CudaParamMap;
use rmlk_schema::{DataType, DataTypeMap};
use std::cmp;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub struct GemmBackend {
    stream: Arc<CudaStream>,
}

impl GemmBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl GemmBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => AddKernel::FwdAlphaBetaF16,
            DataType::Float => AddKernel::FwdAlphaBetaF32,
            DataType::Double => AddKernel::FwdAlphaBetaF64,
            _ => return Err(InternalError::UnsupportedDataType { dtype }.into()),
        };

        add::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    pub fn compute_bias_addition<T>(
        self,
        params: &GemmParams,
        attrs: &GemmAttributes,
        ctx: &Context<Cuda>,
    ) -> Result<()>
    where
        T: CudaParamMap + DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromF32,
    {
        let func = self.load_cuda_function(<T as DataTypeMap>::data_type())?;

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let c = ctx.get_input(2)?;
        let c_payload = c.payload();
        let c_data = c_payload.data::<T>();

        let y = ctx.get_output(0)?;
        let mut y_payload = y.payload_mut();
        let mut y_data = y_payload.data_mut();

        let beta = T::from_f32(attrs.beta());

        let (_, c_stride) = compute_bias_shape(&c.shape(), params)?;

        let rank = y.shape().len();
        let info_on_host = ctx.execution_state().scratch_alloc().allocate(3 * rank)?;
        info_on_host[..rank].copy_from_slice(&y.shape());
        info_on_host[rank..2 * rank].copy_from_slice(&c_stride);
        info_on_host[2 * rank..].copy_from_slice(&y.stride());

        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let info = cuda_bump
            .alloc_from_slice_with_fallback(info_on_host)
            .ok_or(InternalError::CudaBumpAllocatorFailed)?;
        let info_data = info.data::<usize>();

        unsafe {
            binary::compute_alpha_beta_inplace(
                &self.stream,
                func,
                beta,
                T::one(),
                rank,
                &info_data,
                &c_data,
                &mut y_data,
            )?;
        }

        Ok(())
    }

    pub fn compute_multiplication<T>(
        &self,
        params: &GemmParams,
        attrs: &GemmAttributes,
        ctx: &Context<Cuda>,
    ) -> Result<()>
    where
        T: CudaParamMap + DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromF32,
    {
        compute_output_shape(&params, ctx)?;

        let config = create_config(&attrs, &params)?;

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
        y.init_payload::<T>()?;

        debug!(
            "[c][dtype={:?}][shape={:?}][stride=[{:?}]",
            y.dtype(),
            y.shape(),
            y.stride()
        );

        let a_payload = a.payload();
        let a_data = a_payload.data::<T>();

        let b_payload = b.payload();
        let b_data = b_payload.data::<T>();

        let mut y_payload = y.payload_mut();
        let mut y_data = y_payload.data_mut();

        gemm::compute::<T>(&self.stream, &a_data, &b_data, &mut y_data, config)?;

        Ok(())
    }

    pub fn compute_gemm<T>(self, ctx: &Context<Cuda>) -> Result<()>
    where
        T: CudaParamMap + DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromF32,
    {
        let attrs = match ctx.get_attributes() {
            Some(attrs) => GemmAttributes::new(&attrs)?,
            None => GemmAttributes::default(),
        };

        debug!("attributes={:?}", attrs);

        let params = prepare_gemm_params(&attrs, ctx)?;

        self.compute_multiplication::<T>(&params, &attrs, ctx)?;

        if ctx.input_exists(2) {
            self.compute_bias_addition::<T>(&params, &attrs, ctx)?;
        }

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_gemm::<f16>(ctx),
            DataType::Float => self.compute_gemm::<f32>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

fn create_config<T>(
    attrs: &GemmAttributes,
    gemm_params: &GemmParams,
) -> Result<StridedBatchedConfig<T>>
where
    T: CudaParamMap + DataTypeMap + ValidAsZeroBits + DeviceRepr + Num + FromF32,
{
    let alpha = T::from_f32(attrs.alpha());
    gemm::strided_batch_config::<T>((alpha, T::zero()), &gemm_params).map_err(Into::into)
}

fn prepare_gemm_params(attrs: &GemmAttributes, ctx: &Context<Cuda>) -> Result<GemmParams> {
    let a = ctx.get_input(0)?;
    let b = ctx.get_input(1)?;

    let output_ndims = cmp::max(a.shape().len(), b.shape().len());
    if output_ndims != 2 {
        return Err(GemmError::Expected2DInputs {
            a_shape: a.shape().to_vec(),
            b_shape: b.shape().to_vec(),
        }
        .into());
    }

    let res = gemm::gemm_params(
        &a.shape(),
        &a.stride(),
        &b.shape(),
        &b.stride(),
        attrs.trans_a(),
        attrs.trans_b(),
        1,
    )?;

    Ok(res)
}

fn compute_output_shape(params: &GemmParams, ctx: &Context<Cuda>) -> Result<()> {
    let y = ctx.get_output(0)?;
    y.copy_shape_from_slice(&[params.m, params.n]);
    Ok(())
}

fn compute_bias_shape(shape: &[usize], params: &GemmParams) -> Result<([usize; 2], [usize; 2])> {
    match shape.len() {
        1 if params.n == shape[0] => Ok(([1, shape[0]], [0, 1])),
        1 if params.m == shape[0] => Ok(([shape[0], 1], [1, 0])),
        2 if (params.m == shape[0] || shape[0] == 1) && (params.n == shape[1] || shape[1] == 1) => {
            let first_dim_stride = if shape[0] == 1 { 0 } else { shape[0] };
            let second_dim_stride = if shape[1] == 1 { 0 } else { shape[1] };
            Ok(([shape[0], shape[1]], [first_dim_stride, second_dim_stride]))
        }
        2 if (params.m == shape[1] || shape[1] == 1) && (params.n == shape[0] || shape[0] == 1) => {
            let first_dim_stride = if shape[1] == 1 { 0 } else { shape[1] };
            let second_dim_stride = if shape[0] == 1 { 0 } else { shape[0] };
            Ok(([shape[0], shape[1]], [first_dim_stride, second_dim_stride]))
        }
        _ => Err(InternalError::InvalidTensorShape {
            shape: shape.to_vec(),
        }
        .into()),
    }
}

#[derive(Debug)]
pub enum GemmError {
    Expected2DInputs {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
}

impl Display for GemmError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GemmError::Expected2DInputs { a_shape, b_shape } => {
                write!(f, "expected 2d inputs, got {:?} and {:?}", a_shape, b_shape)
            }
        }
    }
}

impl std::error::Error for GemmError {}
