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
use rmlk_cuda::kernels::sub;
use rmlk_cuda::kernels::sub::SubKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct SubBackend {
    stream: Arc<CudaStream>,
}

impl SubBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl SubBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => SubKernel::SubFwdF16,
            DataType::Float => SubKernel::SubFwdF32,
            DataType::Double => SubKernel::SubFwdF64,
            DataType::Int32 => SubKernel::SubFwdI32,
            DataType::Int64 => SubKernel::SubFwdI64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        sub::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_sub<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            binary::compute::<D, D, D>("sub", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_binary::<D, D, D>(
            "debugging/sub",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_sub::<f16>(ctx),
            DataType::Float => self.compute_sub::<f32>(ctx),
            DataType::Double => self.compute_sub::<f64>(ctx),
            DataType::Int32 => self.compute_sub::<i32>(ctx),
            DataType::Int64 => self.compute_sub::<i64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::{assert_close, OpTest};
    use rmlk_schema::Op;

    #[test]
    fn basic() {
        let out = OpTest::new(Op::Sub)
            .input([2, 2], vec![1.0f32, 2.01, 3.0, 4.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.01, 2.0])
            .run::<f32>()
            .unwrap();
        assert_close(&out, &[0.0, 0.00999999, -0.00999999, 2.0]);
    }

    #[test]
    fn i32_inputs() {
        let out = OpTest::new(Op::Sub)
            .input([2], vec![10i32, 5])
            .input([2], vec![3i32, 8])
            .run::<i32>()
            .unwrap();
        assert_eq!(out, vec![7, -3]);
    }

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Sub)
            .input([3], vec![5.0f32, 4.0, 3.0])
            .input([3], vec![1.0f32, 1.0, 1.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![4.0, 3.0, 2.0]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Sub)
            .input([1], vec![true])
            .input([1], vec![false])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
