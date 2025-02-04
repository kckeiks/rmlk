use crate::core::device_service::DeviceService;
use crate::core::error::{InternalError, Result};
use crate::providers::cpu::flatten::FlattenTemplate;
use crate::providers::cuda::activation::ActivationBackend;
use crate::providers::cuda::add::AdditionBackend;
use crate::providers::cuda::conv::ConvolutionBackend;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::gemm::GemmBackend;
use crate::providers::cuda::global_average_pool::GlobalAverageBackend;
use crate::providers::cuda::max_pool::MaxPoolBackend;
use crate::providers::cuda::whereop::WhereBackend;
use crate::providers::cuda::CudaKernel;
use cudarc::driver::{CudaDevice, CudaFunction, DeviceRepr, DriverError};
use rmlk_schema::{DataType, DataTypeMap, Op};
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

    pub fn htod<T>(&self, data: Vec<T>) -> Result<CudaData>
    where
        T: Unpin + DeviceRepr + DataTypeMap,
    {
        let ptr = self.device.htod_copy::<T>(data)?;
        Ok(CudaData::new(ptr))
    }

    pub fn dtoh_float(&self, data: &CudaData) -> Result<Vec<f32>> {
        let ptr = data.data::<f32>();
        Ok(self.device.dtoh_sync_copy::<f32, _>(ptr.as_ref())?)
    }
}

impl DeviceService for Cuda {
    type Data = CudaData;
    type Backend = CudaKernel;

    fn get_backend(&self, op: Op, dtype: DataType) -> Result<Self::Backend> {
        let kernel = match op {
            Op::Add => {
                // Todo: At what point should we load the kernel on device?
                let f = self.load_kernel(op, dtype)?;
                CudaKernel::Add(AdditionBackend::new(self.device.clone(), f))
            }
            Op::Gemm => {
                let f = rmlk_cuda::load_add_kernel_alpha_beta_inplace(&self.device, dtype)?;
                CudaKernel::Gemm(GemmBackend::new(self.device.clone(), f))
            }
            Op::Relu => CudaKernel::Relu(ActivationBackend::new(self.device.clone())),
            Op::Conv => CudaKernel::Conv(ConvolutionBackend::new(self.device.clone())),
            Op::GlobalAveragePool => {
                CudaKernel::GlobalAveragePool(GlobalAverageBackend::new(self.device.clone()))
            }
            Op::MaxPool => CudaKernel::MaxPool(MaxPoolBackend::new(self.device.clone())),
            Op::Flatten => CudaKernel::Flatten(FlattenTemplate::new()),
            Op::Where => {
                let f = self.load_kernel(op, dtype)?;
                CudaKernel::Where(WhereBackend::new(self.device.clone(), f))
            }
            op => {
                return Err(InternalError::UnsupportedOp { op });
            }
        };

        Ok(kernel)
    }

    fn htod_float(&self, data: Vec<f32>) -> Result<CudaData> {
        self.htod(data)
    }

    fn dtoh_float(&self, data: &CudaData) -> Result<Vec<f32>> {
        self.dtoh_float(data)
    }

    fn alloc_zeros_float(&self, len: usize) -> Result<Self::Data> {
        self.device
            .alloc_zeros::<f32>(len)
            .map_err(Into::into)
            .map(CudaData::new)
    }
}

impl From<DriverError> for InternalError {
    fn from(value: DriverError) -> Self {
        rmlk_cuda::Error::Cuda(value.0 as u32).into()
    }
}
