use crate::core::context::Context;
use crate::core::error::{Error, Result};
use crate::core::provider::cuda::data::CudaData;
use cudarc::driver::{CudaDevice, CudaFunction};
use rmlk_ir::DataType;
use std::sync::Arc;

pub struct AddKernel {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl AddKernel {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
    }

    pub fn compute(self, ctx: &mut Context<CudaData>) -> Result<()> {
        let lhs = ctx.get_input(0)?;
        let rhs = ctx.get_input(1)?;

        debug_assert!(lhs.shape() == rhs.shape());

        let elem_count: usize = lhs.shape().iter().product();

        if matches!(lhs.dtype(), &DataType::Float) {
            let lhs_data = lhs
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;
            let rhs_data = rhs
                .data()
                .and_then(|data| data.f32())
                .ok_or(Error::MissingData)?;

            let mut out_slice = unsafe {
                self.device
                    .alloc::<f32>(elem_count)
                    .map_err(|_| Error::AllocationFailed)?
            };

            rmlk_cuda::kernels::add::compute::<f32>(
                self.device,
                self.f,
                lhs_data,
                lhs.shape(),
                lhs.stride(),
                rhs_data,
                rhs.shape(),
                rhs.stride(),
                &mut out_slice,
            )
            .map_err(|_| Error::ComputationFailed)?;

            let result_shape = lhs.shape().clone();
            let result = ctx.get_output_mut(0)?;
            result.init(CudaData::F32(out_slice));
            result._reshape(result_shape);
            result.set_dtype(DataType::Float);
        } else {
            return Err(Error::UnsupportedDataType);
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::context::Context;
    use crate::core::provider::cuda::data::CudaData;
    use crate::core::provider::cuda::kernel::add::AddKernel;
    use crate::core::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_add_f32() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![4, 1, 1, 1];
        let dtype = DataType::Float;

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

        let node_c = TestNode {
            shape,
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a, node_b],
            outputs: vec![node_c],
            attributes: vec![],
            op: Op::Add,
        };

        let mut state = crate::core::test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let f = rmlk_cuda::load_kernel(&device, Op::Add, DataType::Float).unwrap();
        let cuda_kernel = AddKernel::new(device.clone(), f);
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }
}
