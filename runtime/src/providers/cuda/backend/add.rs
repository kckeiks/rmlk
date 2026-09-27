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
use rmlk_cuda::kernels::add;
use rmlk_cuda::kernels::add::AddKernel;
use rmlk_schema::{DataType, DataTypeMap};
use std::sync::Arc;

pub struct AdditionBackend {
    stream: Arc<CudaStream>,
}

impl AdditionBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl AdditionBackend {
    fn load_cuda_function(&self, dtype: DataType) -> Result<CudaFunction> {
        let kernel_name = match dtype {
            DataType::Float16 => AddKernel::FwdF16,
            DataType::Float => AddKernel::FwdF32,
            DataType::Double => AddKernel::FwdF64,
            DataType::Int32 => AddKernel::FwdI32,
            DataType::Int64 => AddKernel::FwdI64,
            _ => return Err(UnsupportedDataType(dtype).into()),
        };

        debug!("[kernel={:?}]", kernel_name);

        add::load_kernel(self.stream.context().clone(), kernel_name).map_err(Into::into)
    }

    fn compute_addition<I>(self, ctx: &Context<Cuda>) -> Result<()>
    where
        I: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
    {
        let cuda_bump = ctx.execution_state().dev().device_allocator().clone();
        let func = self.load_cuda_function(I::data_type())?;
        unsafe {
            binary::compute::<I, I, I>("add", self.stream.clone(), cuda_bump, func, ctx)?;
        }

        #[cfg(feature = "dump")]
        debug::write_results_binary::<I, I, I>(
            "debugging/add",
            self.stream.clone(),
            ctx,
            Default::default(),
        )?;

        Ok(())
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => self.compute_addition::<f16>(ctx),
            DataType::Float => self.compute_addition::<f32>(ctx),
            DataType::Double => self.compute_addition::<f64>(ctx),
            DataType::Int32 => self.compute_addition::<i32>(ctx),
            DataType::Uint32 => self.compute_addition::<u32>(ctx),
            DataType::Int64 => self.compute_addition::<i64>(ctx),
            DataType::Uint64 => self.compute_addition::<u64>(ctx),
            _ => Err(UnsupportedDataType(dtype).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::{assert_close, OpTest};
    use half::f16;
    use rmlk_schema::Op;

    #[test]
    fn two_inputs() {
        let out = OpTest::new(Op::Add)
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn all_scalars() {
        let out = OpTest::new(Op::Add)
            .input([], vec![4.0f32])
            .input([], vec![2.0f32])
            .output([])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![6.0]);
    }

    #[test]
    fn left_scalar() {
        let out = OpTest::new(Op::Add)
            .input([], vec![4.0f32])
            .input([2, 2], vec![2.0f32, 1.0, 10.0, 20.0])
            .output([2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![6.0, 5.0, 14.0, 24.0]);
    }

    #[test]
    fn right_scalar() {
        let out = OpTest::new(Op::Add)
            .input([2, 2], vec![100.0f32, 200.0, 300.0, 400.0])
            .input([], vec![4.0f32])
            .output([2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![104.0, 204.0, 304.0, 404.0]);
    }

    #[test]
    fn one_elem_tensor() {
        let out = OpTest::new(Op::Add)
            .input([2, 2], vec![100.0f32, 200.0, 300.0, 400.0])
            .input([1], vec![4.0f32])
            .output([2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![104.0, 204.0, 304.0, 404.0]);
    }

    #[test]
    fn with_constant() {
        // Former add_with_constants: (a + b) + const, expressed as two Adds.
        let ab = OpTest::new(Op::Add)
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<f32>()
            .unwrap();
        let out = OpTest::new(Op::Add)
            .input([2, 2], ab)
            .constant([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn broadcast_with_constant() {
        // Former add_broadcast: (a + b) + const[1, 2].
        let ab = OpTest::new(Op::Add)
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<f32>()
            .unwrap();
        let out = OpTest::new(Op::Add)
            .input([2, 2], ab)
            .constant([1, 2], vec![3.0f32, 4.0])
            .output([2, 2])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![5.0, 8.0, 9.0, 12.0]);
    }

    #[test]
    fn broadcast_diff_len_shapes() {
        // Former add_broadcast_diff_len_shapes: (a + b) + const[3].
        let ab = OpTest::new(Op::Add)
            .input(
                [2, 4, 1],
                vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            )
            .input(
                [4, 3],
                vec![
                    1.1f32, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1,
                ],
            )
            .output([2, 4, 3])
            .run::<f32>()
            .unwrap();
        let out = OpTest::new(Op::Add)
            .input([2, 4, 3], ab)
            .constant([3], vec![100.0f32, 200.0, 300.0])
            .output([2, 4, 3])
            .run::<f32>()
            .unwrap();
        assert_close(
            &out,
            &[
                111.1, 212.1, 313.1, 124.1, 225.1, 326.1, 137.1, 238.1, 339.1, 150.1, 251.1, 352.1,
                151.1, 252.1, 353.1, 164.1, 265.1, 366.1, 177.1, 278.1, 379.1, 190.1, 291.1, 392.1,
            ],
        );
    }

    #[test]
    fn three_inputs_chained() {
        // Former add_three_inputs: (a + b) + c.
        let ab = OpTest::new(Op::Add)
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<f32>()
            .unwrap();
        let out = OpTest::new(Op::Add)
            .input([2, 2], ab)
            .input([2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![3.0, 6.0, 9.0, 12.0]);
    }

    // --- dtype / shape edge cases ---

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Add)
            .input([4], vec![1.0f32, 2.0, 3.0, 4.0])
            .input([4], vec![10.0f32, 20.0, 30.0, 40.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn i32_inputs() {
        let out = OpTest::new(Op::Add)
            .input([3], vec![1i32, 2, 3])
            .input([3], vec![10i32, 20, 30])
            .run::<i32>()
            .unwrap();
        assert_eq!(out, vec![11, 22, 33]);
    }

    #[test]
    fn i64_inputs() {
        let out = OpTest::new(Op::Add)
            .input([2], vec![2i64, 3])
            .input([2], vec![1i64, 1])
            .run::<i64>()
            .unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    #[ignore = "CudaData::drop panics on Float16 (TODO.md item 21)"]
    fn f16_inputs() {
        let a = [1.0f32, 2.0, 3.0, 4.0].map(f16::from_f32).to_vec();
        let b = [1.0f32, 2.0, 3.0, 4.0].map(f16::from_f32).to_vec();
        let out = OpTest::new(Op::Add)
            .input([2, 2], a)
            .input([2, 2], b)
            .run::<f16>()
            .unwrap();
        let expected = [2.0f32, 4.0, 6.0, 8.0].map(f16::from_f32);
        assert_eq!(out, expected);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Add)
            .input([], vec![true])
            .input([], vec![false])
            .output([])
            .run_err();
        let msg = format!("{err:?}");
        assert!(
            msg.to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
