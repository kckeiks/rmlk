use crate::core::error::UnsupportedDataType;
use crate::core::Context;
use crate::providers::cuda::backend::unary;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::neg;
use rmlk_cuda::kernels::neg::NegKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct NegBackend {
    stream: Arc<CudaStream>,
}

impl NegBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl NegBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => NegKernel::NegFwdF16,
            DataType::Float => NegKernel::NegFwdF32,
            DataType::Double => NegKernel::NegFwdF64,
            DataType::Int32 => NegKernel::NegFwdI32,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        neg::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_neg<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            unary::compute::<D>("neg", self.stream.clone(), func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_unary::<D, D>(
            "debugging/neg",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_neg::<f16>(ctx),
            DataType::Float => self.compute_neg::<f32>(ctx),
            DataType::Double => self.compute_neg::<f64>(ctx),
            DataType::Int32 => self.compute_neg::<i32>(ctx),
            DataType::Int64 => self.compute_neg::<i64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::Op;

    #[test]
    fn basic() {
        let out = OpTest::new(Op::Neg)
            .input([2, 2], vec![1.0f32, -4.0, 0.0, 5.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![-1.0, 4.0, 0.0, -5.0]);
    }

    #[test]
    fn i32_inputs() {
        let out = OpTest::new(Op::Neg)
            .input([3], vec![1i32, -2, 0])
            .run::<i32>()
            .unwrap();
        assert_eq!(out, vec![-1, 2, 0]);
    }

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Neg)
            .input([2], vec![3.0f32, -3.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![-3.0, 3.0]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Neg).input([1], vec![true]).run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
