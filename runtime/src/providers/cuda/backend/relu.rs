use crate::providers::cuda::activation::ActivationKernel;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub struct ReluKernel(());

impl ActivationKernel for ReluKernel {
    fn execute<T>(
        stream: &Arc<CudaStream>,
        alpha: T,
        beta: T,
        x_data: &CudaSlice<T>,
        x_shape: &[i32],
        x_stride: &[i32],
        y_data: &mut CudaSlice<T>,
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::relu::compute(stream, (alpha, beta), x_data, x_shape, x_stride, y_data)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::OpTest;
    use rmlk_schema::Op;

    #[test]
    fn basic() {
        let out = OpTest::new(Op::Relu)
            .input([1, 2, 2, 1], vec![10.0f32, -10.0, 5.0, -5.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![10.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Relu)
            .input([4], vec![-1.0f32, 0.0, 1.0, 2.0])
            .run::<f32>()
            .unwrap();
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Relu).input([1], vec![true]).run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
