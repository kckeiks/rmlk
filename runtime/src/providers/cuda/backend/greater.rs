use crate::core::error::UnsupportedDataType;
use crate::core::Context;
use crate::providers::cuda::backend::binary;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::greater;
use rmlk_cuda::kernels::greater::GreaterKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct GreaterBackend {
    stream: Arc<CudaStream>,
}

impl GreaterBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl GreaterBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => GreaterKernel::GreaterFwdF16,
            DataType::Float => GreaterKernel::GreaterFwdF32,
            DataType::Double => GreaterKernel::GreaterFwdF64,
            DataType::Int32 => GreaterKernel::GreaterFwdI32,
            DataType::Int64 => GreaterKernel::GreaterFwdI64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        greater::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_greater<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            binary::compute::<D, D, bool>("greater", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_binary::<D, D, bool>(
            "debugging/greater",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_greater::<f16>(ctx),
            DataType::Float => self.compute_greater::<f32>(ctx),
            DataType::Double => self.compute_greater::<f64>(ctx),
            DataType::Int32 => self.compute_greater::<i32>(ctx),
            DataType::Int64 => self.compute_greater::<i64>(ctx),
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
        let out = OpTest::new(Op::Greater)
            .input([2, 2], vec![1.0f32, 3.0, 3.001, 1.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<bool>()
            .unwrap();
        assert_eq!(out, vec![false, true, true, false]);
    }

    #[test]
    fn i32_inputs() {
        let out = OpTest::new(Op::Greater)
            .input([3], vec![1i32, 5, 3])
            .input([3], vec![2i32, 4, 3])
            .run::<bool>()
            .unwrap();
        assert_eq!(out, vec![false, true, false]);
    }

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Greater)
            .input([2], vec![2.0f32, 1.0])
            .input([2], vec![1.0f32, 3.0])
            .run::<bool>()
            .unwrap();
        assert_eq!(out, vec![true, false]);
    }

    #[test]
    fn rejects_bool_inputs() {
        let err = OpTest::new(Op::Greater)
            .input([1], vec![true])
            .input([1], vec![false])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
