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
use rmlk_cuda::kernels::div;
use rmlk_cuda::kernels::div::DivKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct DivBackend {
    stream: Arc<CudaStream>,
}

impl DivBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl DivBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => DivKernel::DivFwdF16,
            DataType::Float => DivKernel::DivFwdF32,
            DataType::Double => DivKernel::DivFwdF64,
            DataType::Int32 => DivKernel::DivFwdI32,
            DataType::Int64 => DivKernel::DivFwdI64,
            _ => {
                return Err(UnsupportedDataType(dtype).into());
            }
        };

        debug!("[kernel={:?}]", kernel_name);

        div::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_div<D>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(D::data_type())?;
        unsafe {
            binary::compute::<D, D, D>("div", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_binary::<D, D, D>(
            "debugging/div",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_div::<f16>(ctx),
            DataType::Float => self.compute_div::<f32>(ctx),
            DataType::Double => self.compute_div::<f64>(ctx),
            DataType::Int32 => self.compute_div::<i32>(ctx),
            DataType::Int64 => self.compute_div::<i64>(ctx),
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
        let out = OpTest::new(Op::Div)
            .input([2, 2], vec![1.0f32, 2.0, 9.0, 5.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 2.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![1.0, 1.0, 3.0, 2.5]);
    }

    #[test]
    fn i32_inputs() {
        let out = OpTest::new(Op::Div)
            .input([2], vec![10i32, 9])
            .input([2], vec![2i32, 3])
            .run::<i32>()
            .unwrap();
        assert_eq!(out, vec![5, 3]);
    }

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Div)
            .input([2], vec![8.0f32, 9.0])
            .input([2], vec![2.0f32, 3.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![4.0, 3.0]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Div)
            .input([1], vec![true])
            .input([1], vec![false])
            .run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
