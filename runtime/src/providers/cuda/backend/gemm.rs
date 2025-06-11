use crate::attributes::gemm::GemmAttributes;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::Cuda;
use cudarc::cublas::StridedBatchedConfig;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::gemm::GemmParams;
use rmlk_cuda::kernels::{binary, gemm};
use rmlk_cuda::params::CudaParamMap;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::cmp;
use std::sync::Arc;
use crate::providers::cuda::backend::common;

pub struct GemmBackend {
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl GemmBackend {
    pub fn new(stream: Arc<CudaStream>, func: CudaFunction) -> Self {
        Self { stream, func }
    }
}

impl GemmBackend {
    fn prepare_gemm_params(
        &self,
        attrs: &GemmAttributes,
        ctx: &mut Context<Cuda>,
    ) -> Result<GemmParams> {
        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        let output_ndims = cmp::max(a.shape().len(), b.shape().len());
        if output_ndims != 2 {
            return Err(InternalError::IncompatibleTensorShape {
                shapes: [
                    (a.src_id().into(), a.shape().to_vec()),
                    (b.src_id().into(), b.shape().to_vec()),
                ]
                .try_into()
                .expect("Small map so should succeed"),
                op: Op::Gemm,
            });
        }

        gemm::gemm_params(
            a.shape(),
            a.stride(),
            b.shape(),
            b.stride(),
            attrs.trans_a(),
            attrs.trans_b(),
            1,
        )
        .map_err(Into::into)
    }

    fn compute_output_shape(&self, params: &GemmParams, ctx: &mut Context<Cuda>) -> Result<()> {
        let y = ctx.get_output(0)?;
        let y_index = y.dst_id();
        ctx.execution_state_mut()
            .copy_shape_from_slice(&[params.m, params.n], y_index)?;
        Ok(())
    }

    fn create_config<T>(
        &self,
        attrs: &GemmAttributes,
        gemm_params: &GemmParams,
    ) -> Result<StridedBatchedConfig<T>>
    where
        T: CudaParamMap
            + DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + TryFrom<f32>,
    {
        let alpha = T::try_from(attrs.alpha()).map_err(|_| InternalError::UnableToConvertValue)?;
        gemm::strided_batch_config::<T>((T::from(alpha), T::zero()), &gemm_params)
            .map_err(Into::into)
    }

    pub fn compute_bias_addition<D, T>(
        self,
        params: &GemmParams,
        attrs: &GemmAttributes,
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
        T: GemmKernel,
    {
        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let c = ctx.get_input(2)?;
        let c_dev_data_ref = c.try_dev_data_ptr()?;
        let c_dev_data = c_dev_data_ref.data::<D>();

        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        let beta = D::try_from(attrs.beta()).map_err(|_| InternalError::UnableToConvertValue)?;

        let (_, c_stride) = compute_bias_shape(c.shape(), params)?;

        let rank = y.shape().len();
        let info_buffer = ctx.execution_state().scratch_alloc().allocate(3 * rank)?;
        info_buffer[..rank].copy_from_slice(y.shape());
        info_buffer[rank..2 * rank].copy_from_slice(&c_stride);
        info_buffer[2 * rank..].copy_from_slice(y.stride());

        T::execute_bias_addition::<D>(
            self.func,
            &self.stream,
            rank,
            info_buffer,
            beta,
            &c_dev_data,
            &mut y_dev_data,
        )?;

        Ok(())
    }

    pub fn compute_multiplication<D, T>(
        &self,
        params: &GemmParams,
        attrs: &GemmAttributes,
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
        T: GemmKernel,
    {
        self.compute_output_shape(&params, ctx)?;
        let config = self.create_config(&attrs, &params)?;

        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;
        let y = ctx.get_output(0)?;

        debug!("[a][gemm][shape={:?}][stride=[{:?}]", a.shape(), a.stride());
        debug!("[b][gemm][shape={:?}][stride=[{:?}]", b.shape(), b.stride());
        debug!("[c][gemm][shape={:?}][stride=[{:?}]", y.shape(), y.stride());

        let a_dev_data_ref = a.try_dev_data_ptr()?;
        let a_dev_data = a_dev_data_ref.data::<D>();

        let b_dev_data_ref = b.try_dev_data_ptr()?;
        let b_dev_data = b_dev_data_ref.data::<D>();

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

        T::execute_multiplication::<D>(
            &self.stream,
            &a_dev_data,
            &b_dev_data,
            &mut y_dev_data,
            config,
        )?;

        Ok(())
    }

    pub fn compute_gemm<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: CudaParamMap
            + DataTypeMap
            + CudnnDataType
            + ValidAsZeroBits
            + DeviceRepr
            + Num
            + TryFrom<f32>,
        T: GemmKernel,
    {
        let attrs = match ctx.get_attributes() {
            Some(attrs) => GemmAttributes::new(&attrs)?,
            None => GemmAttributes::default(),
        };

        let params = self.prepare_gemm_params(&attrs, ctx)?;

        self.compute_multiplication::<D, T>(&params, &attrs, ctx)?;

        if ctx.input_exists(2) {
            self.compute_bias_addition::<D, T>(&params, &attrs, ctx)?;
        }

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: GemmKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_gemm::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Conv,
                dtype,
            }),
        }
    }
}

pub trait GemmKernel {
    /// This should compute `AB = 1 * AB + beta * C`.
    fn execute_bias_addition<T>(
        func: CudaFunction,
        stream: &Arc<CudaStream>,
        rank: usize,
        info: &[usize],
        beta: T,
        c_dev_data: &CudaSlice<T>,
        ab_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num;

    fn execute_multiplication<T>(
        stream: &Arc<CudaStream>,
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        y_dev_data: &mut CudaSlice<T>,
        config: StridedBatchedConfig<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl GemmKernel for ActiveKernel {
    fn execute_bias_addition<T>(
        func: CudaFunction,
        stream: &Arc<CudaStream>,
        rank: usize,
        info: &[usize],
        beta: T,
        c_dev_data: &CudaSlice<T>,
        ab_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        unsafe {
            binary::compute_alpha_beta_inplace(
                stream,
                func,
                beta,
                T::one(),
                rank,
                info,
                c_dev_data,
                ab_dev_data,
            )
            .map_err(Into::into)
        }
    }

    fn execute_multiplication<T>(
        stream: &Arc<CudaStream>,
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        y_dev_data: &mut CudaSlice<T>,
        config: StridedBatchedConfig<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        gemm::compute::<T>(stream, a_dev_data, b_dev_data, y_dev_data, config).map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl GemmKernel for NoOpKernel {
    fn execute_bias_addition<T>(
        _: CudaFunction,
        _: &Arc<CudaStream>,
        _: usize,
        _: &[usize],
        _: T,
        _: &CudaSlice<T>,
        _: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        Ok(())
    }

    fn execute_multiplication<T>(
        _: &Arc<CudaStream>,
        _: &CudaSlice<T>,
        _: &CudaSlice<T>,
        _: &mut CudaSlice<T>,
        _: StridedBatchedConfig<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        Ok(())
    }
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
        }),
    }
}
