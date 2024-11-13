use crate::core::device_service::DeviceServiceError;
use crate::core::kernel::{KernelError, Result};
use crate::ops::add::Add;
use crate::providers::cuda::data::CudaData;
use cudarc::driver::{CudaDevice, CudaFunction};
use rmlk_schema::DataType;
use std::sync::Arc;

pub struct AddKernel {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl AddKernel {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
    }
}

impl Add for AddKernel {
    type Data = CudaData;

    fn compute(
        self,
        lhs: Self::Data,
        lhs_shape: &[usize],
        lhs_dtype: DataType,
        rhs: Self::Data,
        _rhs_shape: &[usize],
        _rhs_dtype: DataType,
    ) -> Result<Self::Data> {
        let elem_count: usize = lhs_shape.iter().product();

        if matches!(lhs_dtype, DataType::Float) {
            let lhs_data = lhs.f32().ok_or_else(|| {
                KernelError::Other("expected lhs tensor data to be of type `float32`".to_string())
            })?;
            let rhs_data = rhs.f32().ok_or_else(|| {
                KernelError::Other("expected rhs tensor data to be of type `float32`".to_string())
            })?;

            let mut out_slice = unsafe {
                self.device
                    .alloc::<f32>(elem_count)
                    .map_err(rmlk_cuda::Error::from)
                    .map_err(DeviceServiceError::from)?
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
            )?;

            CudaData::F32(out_slice)
        } else {
            return Err(KernelError::Other(format!(
                "unsupported dtype `{:?}`",
                lhs.dtype()
            )));
        }
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::add::AddKernel;
    use crate::providers::cuda::Cuda;
    use crate::test_utils;
    use crate::test_utils::{TestNode, TestParams};
    use cudarc::driver::CudaDevice;
    use rmlk_schema::{DataType, Op};

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

        let params = TestParams {
            inputs: vec![node_a, node_b],
            op: Op::Add,
            attributes: vec![],
        };

        let mut state = test_utils::build_graph_and_state(Cuda::new(device.clone()), params);
        let mut context = Context::new(&mut state, 3).unwrap();

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
