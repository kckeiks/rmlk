use crate::core::device_service::DeviceService;
use crate::core::error::InternalError;
use crate::providers::cpu::flatten::FlattenTemplate;
use crate::providers::cuda::activation::ActivationBackend;
use crate::providers::cuda::add::AdditionBackend;
use crate::providers::cuda::cast::CastBackend;
use crate::providers::cuda::concat::ConcatBackend;
use crate::providers::cuda::constant::ConstantBackend;
use crate::providers::cuda::constant_of_shape::ConstantOfShapeBackend;
use crate::providers::cuda::conv::ConvolutionBackend;
use crate::providers::cuda::cos::CosBackend;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::div::DivBackend;
use crate::providers::cuda::equal::EqualBackend;
use crate::providers::cuda::expand::ExpandBackend;
use crate::providers::cuda::gather::GatherBackend;
use crate::providers::cuda::gemm::GemmBackend;
use crate::providers::cuda::global_average_pool::GlobalAverageBackend;
use crate::providers::cuda::greater::GreaterBackend;
use crate::providers::cuda::matmul::MatMulBackend;
use crate::providers::cuda::max_pool::MaxPoolBackend;
use crate::providers::cuda::mul::MulBackend;
use crate::providers::cuda::neg::NegBackend;
use crate::providers::cuda::pow::PowBackend;
use crate::providers::cuda::range::RangeBackend;
use crate::providers::cuda::reduce_mean::ReduceMeanBackend;
use crate::providers::cuda::reshape::ReshapeBackend;
use crate::providers::cuda::scatter_nd::ScatterNdBackend;
use crate::providers::cuda::shape::ShapeBackend;
use crate::providers::cuda::sin::SinBackend;
use crate::providers::cuda::slice::SliceBackend;
use crate::providers::cuda::softmax::SoftmaxBackend;
use crate::providers::cuda::sqrt::SqrtBackend;
use crate::providers::cuda::sub::SubBackend;
use crate::providers::cuda::transpose::TransposeBackend;
use crate::providers::cuda::trilu::TriluBackend;
use crate::providers::cuda::unsqueeze::UnsqueezeBackend;
use crate::providers::cuda::whereop::WhereBackend;
use crate::providers::cuda::CudaKernel;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaStream, DeviceRepr};
use half::f16;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct Cuda {
    stream: Arc<CudaStream>,
}

impl Cuda {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }

    pub fn htod<T>(&self, data: Vec<T>) -> Result<CudaData>
    where
        T: Unpin + DeviceRepr + DataTypeMap,
    {
        let ptr = self
            .stream
            .memcpy_stod::<T, _>(&data)
            .map_err(|e| InternalError::Device { error: e.into() })?;
        Ok(CudaData::new(ptr))
    }

    pub fn dtoh<T>(&self, data: &CudaData) -> Result<Vec<T>>
    where
        T: Clone + Default + Unpin + DeviceRepr + DataTypeMap,
    {
        let ptr = data.data::<T>();
        Ok(self
            .stream
            .memcpy_dtov::<T, _>(ptr.as_ref())
            .map_err(|e| InternalError::Device { error: e.into() })?)
    }
}

impl DeviceService for Cuda {
    type Data = CudaData;
    type Backend = CudaKernel;

    fn get_backend(&self, op: Op, _dtype: DataType) -> Result<Self::Backend> {
        let kernel = match op {
            Op::Add => CudaKernel::Add(AdditionBackend::new(self.stream.clone())),
            Op::Cast => CudaKernel::Cast(CastBackend::new(&self.stream)),
            Op::Concat => CudaKernel::Concat(ConcatBackend::new(&self.stream)),
            Op::Constant => CudaKernel::Constant(ConstantBackend::new(&self.stream)),
            Op::ConstantOfShape => {
                CudaKernel::ConstantOfShape(ConstantOfShapeBackend::new(&self.stream))
            }
            Op::Cos => CudaKernel::Cos(CosBackend::new(self.stream.clone())),
            Op::Div => CudaKernel::Div(DivBackend::new(self.stream.clone())),
            Op::Expand => CudaKernel::Expand(ExpandBackend::new(&self.stream)),
            Op::Equal => CudaKernel::Equal(EqualBackend::new(self.stream.clone())),
            Op::Gather => CudaKernel::Gather(GatherBackend::new(self.stream.clone())),
            Op::Gemm => CudaKernel::Gemm(GemmBackend::new(self.stream.clone())),
            Op::Greater => CudaKernel::Greater(GreaterBackend::new(self.stream.clone())),
            Op::Relu => CudaKernel::Relu(ActivationBackend::new(self.stream.clone())),
            Op::Conv => CudaKernel::Conv(ConvolutionBackend::new(self.stream.clone())),
            Op::GlobalAveragePool => {
                CudaKernel::GlobalAveragePool(GlobalAverageBackend::new(self.stream.clone()))
            }
            Op::MaxPool => CudaKernel::MaxPool(MaxPoolBackend::new(self.stream.clone())),
            Op::Mul => CudaKernel::Mul(MulBackend::new(self.stream.clone())),
            Op::Neg => CudaKernel::Neg(NegBackend::new(self.stream.clone())),
            Op::Flatten => CudaKernel::Flatten(FlattenTemplate::new()),
            Op::MatMul => CudaKernel::MatMul(MatMulBackend::new(&self.stream)),
            Op::Pow => CudaKernel::Pow(PowBackend::new(self.stream.clone())),
            Op::ReduceMean => CudaKernel::ReduceMean(ReduceMeanBackend::new(self.stream.clone())),
            Op::Range => CudaKernel::Range(RangeBackend::new(&self.stream)),
            Op::Reshape => CudaKernel::Reshape(ReshapeBackend::new(self.stream.clone())),
            Op::ScatterND => CudaKernel::ScatterNd(ScatterNdBackend::new(&self.stream)),
            Op::Shape => CudaKernel::Shape(ShapeBackend::new(&self.stream)),
            Op::Sin => CudaKernel::Sin(SinBackend::new(self.stream.clone())),
            Op::Sigmoid => CudaKernel::Sigmoid(ActivationBackend::new(self.stream.clone())),
            Op::Slice => CudaKernel::Slice(SliceBackend::new(&self.stream)),
            Op::Softmax => CudaKernel::Softmax(SoftmaxBackend::new(&self.stream)),
            Op::Sqrt => CudaKernel::Sqrt(SqrtBackend::new(&self.stream)),
            Op::Sub => CudaKernel::Sub(SubBackend::new(self.stream.clone())),
            Op::Transpose => CudaKernel::Transpose(TransposeBackend::new(self.stream.clone())),
            Op::Trilu => CudaKernel::Trilu(TriluBackend::new(&self.stream)),
            Op::Unsqueeze => CudaKernel::Unsqueeze(UnsqueezeBackend::new(&self.stream)),
            Op::Where => CudaKernel::Where(WhereBackend::new(self.stream.clone())),
            op => {
                return Err(anyhow!("no backend for op `{op:?}`"));
            }
        };

        Ok(kernel)
    }

    fn htod_float16(&self, data: Vec<f16>) -> Result<Self::Data> {
        self.htod(data)
    }

    fn htod_float(&self, data: Vec<f32>) -> Result<CudaData> {
        self.htod(data)
    }

    fn htod_double(&self, data: Vec<f64>) -> Result<CudaData> {
        self.htod(data)
    }

    fn dtoh_float16(&self, data: &Self::Data) -> Result<Vec<f16>> {
        self.dtoh(data)
    }

    fn dtoh_float(&self, data: &CudaData) -> Result<Vec<f32>> {
        self.dtoh(data)
    }

    fn dtoh_i32(&self, data: &Self::Data) -> Result<Vec<i32>> {
        self.dtoh(data)
    }

    fn dtoh_i64(&self, data: &Self::Data) -> Result<Vec<i64>> {
        self.dtoh(data)
    }

    fn dtoh_bool(&self, data: &Self::Data) -> Result<Vec<bool>> {
        self.dtoh(data)
    }

    fn htod_i32(&self, data: Vec<i32>) -> Result<Self::Data> {
        self.htod(data)
    }

    fn htod_i64(&self, data: Vec<i64>) -> Result<Self::Data> {
        self.htod(data)
    }

    fn htod_bool(&self, data: Vec<bool>) -> Result<Self::Data> {
        self.htod(data)
    }

    fn alloc_zeros_float(&self, len: usize) -> Result<Self::Data> {
        self.stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| InternalError::Device { error: e.into() })
            .map_err(Into::into)
            .map(CudaData::new)
    }
}
