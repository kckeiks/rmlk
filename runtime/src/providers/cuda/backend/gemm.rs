use crate::attributes::gemm::GemmAttributes;
use crate::core::allocators::ScratchAllocator;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cublas::StridedBatchedConfig;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::gemm::GemmParams;
use rmlk_cuda::kernels::{binary, gemm};
use rmlk_cuda::params::CudaParamMap;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::cmp;
use std::sync::Arc;

pub struct GemmBackend {
    device: Arc<CudaDevice>,
    func: CudaFunction,
}

impl GemmBackend {
    pub fn new(device: Arc<CudaDevice>, func: CudaFunction) -> Self {
        Self { device, func }
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
            self.device.clone(),
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

        let output_size = y.shape().iter().product();

        let a_dev_data_ref = a.try_dev_data_ptr()?;
        let a_dev_data = a_dev_data_ref.data::<D>();

        let b_dev_data_ref = b.try_dev_data_ptr()?;
        let b_dev_data = b_dev_data_ref.data::<D>();

        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let mut c = ctx.get_output(0)?;
            let mut c_dev_data_ref = c.dev_data_ptr_mut();
            let need_to_alloc_dev_data = c_dev_data_ref.is_none()
                || c_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != output_size)
                    .unwrap_or(true);

            if !need_to_alloc_dev_data {
                c_dev_data_ref.as_mut().expect("").zero::<D>()?
            }

            // We need to remove this immutable reference so we can mutate `y`.
            drop(c_dev_data_ref);

            if need_to_alloc_dev_data {
                let c_dev_data = self
                    .device
                    .alloc_zeros::<D>(output_size)
                    .map_err(rmlk_cuda::Error::from)?;
                c.set_dev_data(CudaData::new(c_dev_data));
            };
        }

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let y = ctx.get_output(0)?;
        let mut y_dev_data_ref = y.dev_data_ptr_mut();
        let mut y_dev_data = y_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        T::execute_multiplication::<D>(
            self.device.clone(),
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
            Some(attrs) => GemmAttributes::new(attrs)?,
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
        device: Arc<CudaDevice>,
        rank: usize,
        info: &[usize],
        beta: T,
        c_dev_data: &CudaSlice<T>,
        ab_dev_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num;

    fn execute_multiplication<T>(
        device: Arc<CudaDevice>,
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
        device: Arc<CudaDevice>,
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
                device,
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
        device: Arc<CudaDevice>,
        a_dev_data: &CudaSlice<T>,
        b_dev_data: &CudaSlice<T>,
        y_dev_data: &mut CudaSlice<T>,
        config: StridedBatchedConfig<T>,
    ) -> Result<()>
    where
        T: CudaParamMap + CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        gemm::compute::<T>(device, a_dev_data, b_dev_data, y_dev_data, config).map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl GemmKernel for NoOpKernel {
    fn execute_bias_addition<T>(
        _: CudaFunction,
        _: Arc<CudaDevice>,
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
        _: Arc<CudaDevice>,
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

// #[cfg(test)]
// mod test {
//     use crate::core::Context;
//     use crate::providers::cuda::data::CudaData;
//     use crate::providers::cuda::kernel::gemm::BackendHandler;
//     use crate::providers::cuda::Cuda;
//     use crate::test_utils;
//     use crate::test_utils::{TestNode, TestParams};
//     use cudarc::driver::CudaDevice;
//     use rmlk_schema::{DataType, Op};
//
//     #[test]
//     fn test_gemm_f32() {
//         let device = CudaDevice::new(0).unwrap();
//
//         let shape = vec![1, 2, 2];
//         let dtype = DataType::Float;
//         let op = Op::Gemm;
//
//         let node_a = TestNode {
//             shape: shape.clone(),
//             dtype,
//             data: Some(CudaData::F32(
//                 device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
//             )),
//         };
//         let node_b = TestNode {
//             shape: shape.clone(),
//             dtype,
//             data: Some(CudaData::F32(
//                 device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
//             )),
//         };
//
//         let params = TestParams {
//             inputs: vec![node_a, node_b],
//             attributes: vec![],
//             op,
//         };
//
//         let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
//         let mut context = Context::new(&mut state, 3).unwrap();
//
//         let cuda_kernel = BackendHandler::new(device.clone());
//         cuda_kernel.compute(&mut context).unwrap();
//
//         let out_data = context
//             .get_output(0)
//             .unwrap()
//             .dev_data_ptr()
//             .unwrap()
//             .f32()
//             .unwrap();
//         let result = device.dtoh_sync_copy(out_data).unwrap();
//
//         assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
//     }
// }
