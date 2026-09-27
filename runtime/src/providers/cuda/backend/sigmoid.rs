use crate::providers::cuda::activation::ActivationKernel;
use anyhow::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

pub struct SigmoidKernel(());

impl ActivationKernel for SigmoidKernel {
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
        rmlk_cuda::kernels::sigmoid::compute(
            stream,
            (alpha, beta),
            x_data,
            x_shape,
            x_stride,
            y_data,
        )
        .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::{assert_close, OpTest};
    use rmlk_schema::Op;

    #[test]
    fn basic_4d() {
        let out = OpTest::new(Op::Sigmoid)
            .input([2, 2, 1, 1], vec![0.5f32, -0.5, 2.0, -2.0])
            .run::<f32>()
            .unwrap();
        assert_close(&out, &[0.62245935, 0.37754068, 0.880797, 0.11920292]);
    }

    #[test]
    fn basic_3d() {
        let out = OpTest::new(Op::Sigmoid)
            .input([2, 2, 1], vec![0.5f32, -0.5, 2.0, -2.0])
            .run::<f32>()
            .unwrap();
        assert_close(&out, &[0.62245935, 0.37754068, 0.880797, 0.11920292]);
    }

    #[test]
    fn rank1() {
        let out = OpTest::new(Op::Sigmoid)
            .input([1], vec![0.0f32])
            .run::<f32>()
            .unwrap();
        assert_close(&out, &[0.5]);
    }

    #[test]
    fn rejects_bool() {
        let err = OpTest::new(Op::Sigmoid).input([1], vec![true]).run_err();
        assert!(
            format!("{err:?}").to_lowercase().contains("unsupported"),
            "unexpected error: {err:?}"
        );
    }
}
