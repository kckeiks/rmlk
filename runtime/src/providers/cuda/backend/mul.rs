use crate::core::error::UnsupportedDataType;
use crate::core::Context;
use crate::providers::cuda::backend::binary;
#[cfg(feature = "dump")]
use crate::providers::cuda::debug;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_cuda::kernels::mul;
use rmlk_cuda::kernels::mul::MulKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct MulBackend {
    stream: Arc<CudaStream>,
}

impl MulBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl MulBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => MulKernel::FwdF16,
            DataType::Float => MulKernel::FwdF32,
            DataType::Double => MulKernel::FwdF64,
            DataType::Int32 => MulKernel::FwdI32,
            DataType::Int64 => MulKernel::FwdI64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        mul::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_mul<I>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(I::data_type())?;
        unsafe {
            binary::compute::<I, I, I>("mul", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_binary::<I, I, I>(
            "debugging/mul",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_mul::<f16>(ctx),
            DataType::Float => self.compute_mul::<f32>(ctx),
            DataType::Double => self.compute_mul::<f64>(ctx),
            DataType::Int32 => self.compute_mul::<i32>(ctx),
            DataType::Int64 => self.compute_mul::<i64>(ctx),
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
        let out = OpTest::new(Op::Mul)
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn i32_inputs() {
        let out = OpTest::new(Op::Mul)
            .input([2], vec![2i32, 3])
            .input([2], vec![4i32, 5])
            .run::<i32>()
            .unwrap();
        assert_eq!(out, vec![8, 15]);
    }

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Mul)
            .input([3], vec![2.0f32, 3.0, 4.0])
            .input([3], vec![10.0f32, 10.0, 10.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Mul)
            .input([1], vec![true])
            .input([1], vec![false])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
