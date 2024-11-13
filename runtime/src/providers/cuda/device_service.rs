use crate::core::device_service::{DeviceService, DeviceServiceError, Result};
use crate::ops::activation::ActivationOp;
use crate::ops::add::AddOp;
use crate::ops::conv::ConvolutionOp;
use crate::ops::flatten::FlattenOp;
use crate::providers::cuda::activation::ActivationKernel;
use crate::providers::cuda::conv::ConvKernel;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::gemm::GemmKernel;
use crate::providers::cuda::global_average_pool::GlobalAveragePoolKernel;
use crate::providers::cuda::kernel::add::AddKernel;
use crate::providers::cuda::max_pool::MaxPoolKernel;
use crate::providers::cuda::CudaKernel;
use cudarc::driver::{CudaDevice, CudaFunction, DriverError};
use rmlk_schema::{DataType, Op};
use std::sync::Arc;

pub struct Cuda {
    device: Arc<CudaDevice>,
}

impl Cuda {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    fn load_kernel(&self, op: Op, dtype: DataType) -> Result<CudaFunction> {
        Ok(rmlk_cuda::load_kernel(&self.device, op, dtype)?)
    }

    pub fn htod_float(&self, data: Vec<f32>) -> Result<CudaData> {
        let ptr = self.device.htod_copy(data)?;
        Ok(CudaData::F32(ptr))
    }

    pub fn dtoh_float(&self, data: &CudaData) -> Result<Vec<f32>> {
        match data {
            CudaData::F32(ptr) => Ok(self.device.dtoh_sync_copy::<f32, _>(ptr)?),
            _ => Err(DeviceServiceError::Other(
                "unsupported data type".to_string(),
            )),
        }
    }
}

impl DeviceService for Cuda {
    type Data = CudaData;
    type Kernel = CudaKernel;

    fn get_kernel(&self, op: Op, dtype: DataType) -> Result<Self::Kernel> {
        let kernel = match op {
            Op::Add => {
                // Todo: At what point should we load the kernel on device?
                let f = self.load_kernel(op, dtype)?;
                CudaKernel::Add(AddOp::new(AddKernel::new(self.device.clone(), f)))
            }
            Op::Gemm => CudaKernel::Gemm(GemmKernel::new(self.device.clone())),
            Op::Relu => CudaKernel::Relu(ActivationOp::new(ActivationKernel::new(
                self.device.clone(),
            ))),
            Op::Conv => CudaKernel::Conv(ConvolutionOp::new(ConvKernel::new(self.device.clone()))),
            Op::GlobalAveragePool => {
                CudaKernel::GlobalAveragePool(GlobalAveragePoolKernel::new(self.device.clone()))
            }
            Op::MaxPool => CudaKernel::MaxPool(MaxPoolKernel::new(self.device.clone())),
            Op::Flatten => CudaKernel::Flatten(FlattenOp::new()),
            op => {
                return Err(DeviceServiceError::Other(format!(
                    "no support for op `{op:?}`"
                )));
            }
        };

        Ok(kernel)
    }

    fn htod_float(&self, data: Vec<f32>) -> Result<CudaData> {
        self.htod_float(data)
    }

    fn dtoh_float(&self, data: &CudaData) -> Result<Vec<f32>> {
        self.dtoh_float(data)
    }

    fn alloc_zeros_float(&self, len: usize) -> Result<Self::Data> {
        self.device
            .alloc_zeros(len)
            .map_err(|e| DeviceServiceError::Cuda(e.into()))
            .map(CudaData::F32)
    }
}

impl From<DriverError> for DeviceServiceError {
    fn from(value: DriverError) -> Self {
        rmlk_cuda::Error::Cuda(value.0 as u32).into()
    }
}
