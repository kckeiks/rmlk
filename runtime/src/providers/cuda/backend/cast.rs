use crate::attributes::cast;
use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::backend::common;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, ValidAsZeroBits};
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct CastBackend {
    device: Arc<CudaDevice>,
}

impl CastBackend {
    pub fn new(device: &Arc<CudaDevice>) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn compute_cast<I, O, K>(
        &mut self,
        kernel: CudaFunction,
        ctx: &mut Context<Cuda>,
    ) -> Result<()>
    where
        I: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        O: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        K: CastKernel,
    {
        let input = ctx.get_input(0)?;

        {
            let output = ctx.get_output(0)?;
            debug!(
                "[input][cast][shape={:?}][stride=[{:?}]",
                input.shape(),
                input.stride()
            );
            debug!(
                "[output][cast][shape={:?}][stride=[{:?}]",
                output.shape(),
                output.stride()
            );
        }

        common::init_tensor_device_data::<I>(&self.device, input)?;

        let input = ctx.get_input(0)?;

        let input_dev_data_ref = input.try_dev_data_ptr()?;
        let input_dev_data = input_dev_data_ref.data::<I>();

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let output = ctx.get_output(0)?;
        let mut output_dev_data_ref = output.dev_data_ptr_mut();
        let mut output_dev_data = output_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        let rank = output.shape().len();

        let info_buffer = ctx.execution_state().scratch_alloc().allocate(2 * rank)?;
        info_buffer[..rank].copy_from_slice(output.shape());
        info_buffer[rank..2 * rank].copy_from_slice(input.stride());

        K::execute::<I, O>(kernel, &input_dev_data, &mut output_dev_data)?;

        Ok(())
    }

    pub fn compute<K>(mut self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        K: CastKernel,
    {
        let in_dtype = ctx.get_input(0)?.dtype();
        let out_dtype = ctx
            .get_attributes()
            .map(cast::get_value)
            .transpose()
            .map(Option::flatten)?
            .ok_or_else(|| InternalError::MissingAttribute {
                name: "missing `to` cast attribute".to_string(),
            })?;

        match (in_dtype, out_dtype) {
            (DataType::Float, DataType::Int32) => {
                // Todo: it would be better to load this before running inference.
                let kernel = rmlk_cuda::load_cast_kernel(
                    &self.device,
                    rmlk_cuda::kernels::cast::CastKernel::F32ToI32,
                )?;
                self.compute_cast::<f32, i32, K>(kernel, ctx)
            }
            _ => Err(InternalError::UnsupportedOpForDataType {
                op: Op::Cast,
                // Todo: fix because we're missing info here.
                dtype: in_dtype,
            }),
        }
    }
}

pub trait CastKernel {
    fn execute<I, O>(
        kernel: CudaFunction,
        input_dev_data: &CudaSlice<I>,
        output_dev_data: &mut CudaSlice<O>,
    ) -> Result<()>
    where
        I: CudnnDataType + ValidAsZeroBits + DeviceRepr,
        O: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl CastKernel for ActiveKernel {
    fn execute<I, O>(
        kernel: CudaFunction,
        input_dev_data: &CudaSlice<I>,
        output_dev_data: &mut CudaSlice<O>,
    ) -> Result<()>
    where
        I: CudnnDataType + ValidAsZeroBits + DeviceRepr,
        O: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        unsafe {
            rmlk_cuda::kernels::unary::explicit_io_types_compute(
                kernel,
                input_dev_data,
                output_dev_data,
            )
            .map_err(Into::into)
        }
    }
}
