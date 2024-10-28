use crate::attributes::gemm::GemmAttributes;
use crate::core::Context;
use crate::core::{Error, Result};
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::driver::CudaDevice;
use log::trace;
use rmlk_cuda::kernels::gemm::GemmOp;
use rmlk_schema::DataType;
use std::sync::Arc;

pub struct GemmKernel {
    device: Arc<CudaDevice>,
}

impl GemmKernel {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let lhs = ctx.get_input(0)?;
        let rhs = ctx.get_input(1)?;

        let attrs = GemmAttributes::new(ctx.get_attributes().ok_or(Error::MissingAttributes)?)?;

        let op = GemmOp::new(
            lhs.shape(),
            lhs.stride(),
            rhs.shape(),
            rhs.stride(),
            attrs.trans_a(),
            attrs.trans_b(),
        );
        let output_size = op.calculate_output_shape().iter().product();

        trace!(
            "lhs_dtype={:?},\
            lhs_shape={:?},\
            lhs_stride={:?},\
            rhs_shape={:?},\
            rhs_stride={:?},\
            trans_a={:?},\
            trans_b={:?}",
            lhs.dtype(),
            lhs.shape(),
            lhs.stride(),
            rhs.shape(),
            rhs.stride(),
            attrs.trans_a(),
            attrs.trans_b(),
        );

        if matches!(lhs.dtype(), DataType::Float) {
            let lhs_data = lhs
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;
            let rhs_data = rhs
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;

            let mut out_slice = self.device.alloc_zeros(output_size)?;

            let config = op
                .strided_batch_config((attrs.alpha(), attrs.beta()))
                .map_err(|_| Error::ComputationFailed)?;

            op.compute_f32(self.device, lhs_data, rhs_data, &mut out_slice, config)
                .map_err(|_| Error::ComputationFailed)?;

            let output = ctx.get_output_mut(0)?;
            output.init(CudaData::F32(out_slice));
            output._reshape(op.calculate_output_shape().to_vec());
            output.set_dtype(DataType::Float);
        } else {
            return Err(Error::UnsupportedDataType);
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::gemm::GemmKernel;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_gemm_f32() {
        let device = CudaDevice::new(0).unwrap();

        let shape = vec![1, 2, 2];
        let dtype = DataType::Float;
        let op = Op::Gemm;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F32(
                device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            attributes: vec![],
            op,
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 3).unwrap();

        let cuda_kernel = GemmKernel::new(device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    }
}
