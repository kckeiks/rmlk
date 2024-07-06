use crate::cuda;
use crate::cuda::data::CudaData;
use crate::cuda::kernel::CudaKernel;
use crate::cuda::kernels::{add, mul};
use crate::error::Error;
use crate::provider::Provider;
use crate::tensor::Tensor;
use crate::Result;
use cudarc::cublas::CudaBlas;
use cudarc::cudnn;
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use half::f16;
use rmlk_ir::{DataType, Op};
use std::sync::Arc;

pub struct CudaProvider {
    device: Arc<CudaDevice>,
    _cudnn: Arc<cudnn::Cudnn>,
}

impl Clone for CudaProvider {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            // Todo: We should be careful here.
            // Two threads cannot use a handle simultaneously,
            // See https://docs.nvidia.com/deeplearning/cudnn/latest/developer/misc.html?highlight=thread%20safety#thread-safety.
            _cudnn: self._cudnn.clone(),
        }
    }
}

impl CudaProvider {
    #[cfg(test)]
    pub fn new(device: Arc<CudaDevice>) -> Self {
        // Todo: Handle the unwrap.
        Self {
            _cudnn: cudnn::Cudnn::new(device.clone()).unwrap(),
            device,
        }
    }

    #[cfg(test)]
    fn htod_f16(&self, data: Vec<f16>) -> Result<CudaData> {
        self.device
            .htod_copy(data)
            .map(CudaData::F16)
            .map_err(|_| Error::Unknown)
    }

    #[cfg(test)]
    fn htod_f32(&self, data: Vec<f32>) -> Result<CudaData> {
        self.device
            .htod_copy(data)
            .map(CudaData::F32)
            .map_err(|_| Error::Unknown)
    }

    #[cfg(test)]
    fn dtoh_f16(&self, data: &CudaData) -> Result<Vec<f16>> {
        let data = data.f16().map_err(|_| Error::Unknown)?;
        self.device.dtoh_sync_copy(data).map_err(|_| Error::Unknown)
    }

    #[cfg(test)]
    fn dtoh_f32(&self, data: &CudaData) -> Result<Vec<f32>> {
        let data = data.f32().map_err(|_| Error::Unknown)?;
        self.device.dtoh_sync_copy(data).map_err(|_| Error::Unknown)
    }

    // pub fn convnd(&self,
    //               input: &Tensor<Self>,
    //               filter: &Tensor<Self>,
    //               output: &mut Tensor<Self>,
    // ) -> Result<()> {
    //         self.cudnn.create_nd_tensor(input.data(), &[])?;
    //     Ok(())
    // }

    pub fn _matmul(
        &self,
        lhs: &Tensor<CudaData>,
        rhs: &Tensor<CudaData>,
        out: &mut Tensor<CudaData>,
    ) -> Result<()> {
        let lhs_shape = lhs.shape();
        let b = lhs_shape[..lhs_shape.len() - 2].iter().product::<usize>();
        let m = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];

        let rhs_shape = rhs.shape();
        let n = rhs_shape[rhs_shape.len() - 2];

        match *lhs.dtype() {
            DataType::Float16 => {
                todo!()
            }
            DataType::Float => {
                let lhs_stride = lhs.stride();
                let rhs_stride = rhs.stride();
                let config = cuda::ops::gemm::gemm_config::<f32>(
                    1.0,
                    0.0,
                    (b, m, n, k),
                    (lhs_shape, lhs_stride),
                    (rhs_shape, rhs_stride),
                )
                .unwrap();
                let mut out_slice = unsafe { self.device.alloc::<f32>(b * m * n).unwrap() };
                let cublas = CudaBlas::new(self.device.clone()).unwrap();
                unsafe {
                    cuda::ops::gemm::gemm_stride_batched_f32(
                        &cublas,
                        config,
                        &rhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                        &lhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                        &mut out_slice,
                    )
                    .unwrap();
                };
                let _ = out.init(CudaData::F32(out_slice));
            }
            DataType::Double => {
                todo!()
            }
            _ => todo!(),
        }
        Ok(())
    }

    pub fn _kernel(&self, op: Op, dtype: DataType) -> Result<CudaFunction> {
        let (fwd_fn_name, fwd_fn_all, module_name, ptx_src) = match op {
            Op::Add => (
                add::FWD_FN_NAMES[dtype as usize],
                add::FWD_FN_NAMES.as_slice(),
                add::MODULE_NAME,
                add::PTX_SRC,
            ),
            Op::Mul => (
                mul::FWD_FN_NAMES[dtype as usize],
                mul::FWD_FN_NAMES.as_slice(),
                mul::MODULE_NAME,
                mul::PTX_SRC,
            ),
            _ => unimplemented!(),
        };

        if !self.device.has_func(module_name, fwd_fn_name) {
            self.device
                .load_ptx(ptx_src.into(), module_name, fwd_fn_all)
                .map_err(|_| Error::Unknown)?
        }
        Ok(self
            .device
            .get_func(module_name, fwd_fn_name)
            .expect("To have been loaded"))
    }

    pub fn _forward(
        &self,
        op: Op,
        lhs: &Tensor<CudaData>,
        rhs: &Tensor<CudaData>,
        out: &mut Tensor<CudaData>,
    ) -> Result<()> {
        // Todo: more assertions here.
        debug_assert!(lhs.shape() == rhs.shape());

        // Todo: Validate that the tensors are valid for the operation.

        let func = self._kernel(op, *lhs.dtype())?;

        // Todo: should we directly initialize this in the device?
        let mut info: Vec<usize> = Vec::with_capacity(3 * lhs.shape().len());
        info.extend(lhs.shape());
        info.extend(lhs.stride());
        info.extend(rhs.stride());
        let info = self
            .device
            .htod_copy(info.as_slice().to_vec())
            .map_err(|_| Error::Unknown)?;

        let elem_count: usize = lhs.shape().iter().product();

        let num_threads = 128;
        let num_blocks = (elem_count + num_threads - 1) / num_threads;
        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if matches!(lhs.dtype(), &DataType::Float16) {
            let lhs_data = lhs.data().ok_or(Error::Executor)?.f16()?;
            let rhs_data = rhs.data().ok_or(Error::Executor)?.f16()?;
            let mut out_slice = unsafe { self.device.alloc::<f16>(elem_count).unwrap() };

            let params = (
                elem_count,
                lhs.shape().len(),
                &info,
                lhs_data,
                rhs_data,
                &mut out_slice,
            );
            unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };
            let _ = out.init(CudaData::F16(out_slice));
        } else if matches!(lhs.dtype(), &DataType::Float) {
            let lhs_data = lhs.data().ok_or(Error::Executor)?.f32()?;
            let rhs_data = rhs.data().ok_or(Error::Executor)?.f32()?;
            let mut out_slice = unsafe { self.device.alloc::<f32>(elem_count).unwrap() };

            let params = (
                elem_count,
                lhs.shape().len(),
                &info,
                lhs_data,
                rhs_data,
                &mut out_slice,
            );
            unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };
            let _ = out.init(CudaData::F32(out_slice));
        } else if matches!(lhs.dtype(), &DataType::Double) {
            let lhs_data = rhs.data().ok_or(Error::Executor)?.f64()?;
            let rhs_data = lhs.data().ok_or(Error::Executor)?.f64()?;
            let mut out_slice = unsafe { self.device.alloc::<f64>(elem_count).unwrap() };

            let params = (
                elem_count,
                lhs.shape().len(),
                &info,
                lhs_data,
                rhs_data,
                &mut out_slice,
            );
            unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };
            let _ = out.init(CudaData::F64(out_slice));
        }

        Ok(())
    }
}

impl Provider for CudaProvider {
    type Kernel = CudaKernel;

    fn kernel(&self, op: Op) -> Self::Kernel {
        CudaKernel::new(op, self.device.clone())
    }
}

#[cfg(test)]
mod test {
    use crate::cuda::provider::CudaProvider;
    use crate::tensor::Tensor;
    use cudarc::driver::CudaDevice;
    use half::f16;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_add_f16() {
        let device = CudaDevice::new(0).unwrap();
        let cuda = CudaProvider::new(device);

        let shape = [4, 1, 1, 1];

        let mut lhs_strides = [0usize; 4];
        lhs_strides[0] = 1;
        for i in 1..4 {
            lhs_strides[i] += lhs_strides[i - 1] * shape[i - 1];
        }
        let mut rhs_strides = [0usize; 4];
        rhs_strides[0] = 1;
        for i in 1..4 {
            rhs_strides[i] += rhs_strides[i - 1] * shape[i - 1];
        }

        let mut lhs_tensor = Tensor::new(DataType::Float16, shape.to_vec());
        let mut out_tensor = Tensor::new(DataType::Float16, shape.to_vec());
        let mut rhs_tensor = Tensor::new(DataType::Float16, shape.to_vec());

        let lhs_data = cuda
            .htod_f16(vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ])
            .unwrap();
        let rhs_data = cuda
            .htod_f16(vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ])
            .unwrap();

        lhs_tensor.init(lhs_data);
        rhs_tensor.init(rhs_data);

        cuda._forward(Op::Add, &lhs_tensor, &rhs_tensor, &mut out_tensor)
            .unwrap();
        let result = cuda.dtoh_f16(out_tensor.data().unwrap()).unwrap();
        assert_eq!(
            result,
            vec![
                f16::from_f32(2.0),
                f16::from_f32(4.0),
                f16::from_f32(6.0),
                f16::from_f32(8.0),
            ]
        )
    }

    #[test]
    fn test_add_f32() {
        let device = CudaDevice::new(0).unwrap();
        let cuda = CudaProvider::new(device);

        let shape = [4, 1, 1, 1];

        let mut lhs_strides = [0usize; 4];
        lhs_strides[0] = 1;
        for i in 1..4 {
            lhs_strides[i] += lhs_strides[i - 1] * shape[i - 1];
        }
        let mut rhs_strides = [0usize; 4];
        rhs_strides[0] = 1;
        for i in 1..4 {
            rhs_strides[i] += rhs_strides[i - 1] * shape[i - 1];
        }

        let mut lhs_tensor = Tensor::new(DataType::Float, shape.to_vec());
        let mut rhs_tensor = Tensor::new(DataType::Float, shape.to_vec());
        let mut out_tensor = Tensor::new(DataType::Float, shape.to_vec());

        let lhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
        let rhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();

        lhs_tensor.init(lhs_data);
        rhs_tensor.init(rhs_data);

        cuda._forward(Op::Add, &lhs_tensor, &rhs_tensor, &mut out_tensor)
            .unwrap();
        let result = cuda.dtoh_f32(out_tensor.data().unwrap()).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }

    #[test]
    fn test_mul_f16() {
        let device = CudaDevice::new(0).unwrap();
        let cuda = CudaProvider::new(device);

        let shape = [3, 4, 5];
        let elem_num = shape.iter().product::<usize>();
        let dims = shape.len();

        let mut lhs_strides = vec![0usize; dims];
        lhs_strides[0] = 1;
        for i in 1..dims {
            lhs_strides[i] += lhs_strides[i - 1] * shape[i - 1];
        }
        let mut rhs_strides = vec![0usize; dims];
        rhs_strides[0] = 1;
        for i in 1..dims {
            rhs_strides[i] += rhs_strides[i - 1] * shape[i - 1];
        }

        let mut lhs_tensor = Tensor::new(DataType::Float16, shape.to_vec());
        let mut rhs_tensor = Tensor::new(DataType::Float16, shape.to_vec());
        let mut out_tensor = Tensor::new(DataType::Float16, shape.to_vec());

        let lhs_data = cuda.htod_f16(vec![f16::from_f32(2.0); elem_num]).unwrap();
        let rhs_data = cuda.htod_f16(vec![f16::from_f32(3.0); elem_num]).unwrap();

        lhs_tensor.init(lhs_data);
        rhs_tensor.init(rhs_data);

        cuda._forward(Op::Mul, &lhs_tensor, &rhs_tensor, &mut out_tensor)
            .unwrap();
        let result = cuda.dtoh_f16(out_tensor.data().unwrap()).unwrap();
        assert_eq!(result, vec![f16::from_f32(6.0); elem_num])
    }

    #[test]
    fn test_matmul_f32() {
        let device = CudaDevice::new(0).unwrap();
        let cuda = CudaProvider::new(device);

        let shape = [1, 2, 2];

        let mut lhs_strides = [0usize; 3];
        lhs_strides[2] = 1;
        for i in (0..2).rev() {
            lhs_strides[i] += lhs_strides[i + 1] * shape[i + 1];
        }
        let mut rhs_strides = [0usize; 3];
        rhs_strides[2] = 1;
        for i in (0..2).rev() {
            rhs_strides[i] += rhs_strides[i + 1] * shape[i + 1];
        }

        let mut lhs_tensor = Tensor::new(DataType::Float, shape.to_vec());
        let mut rhs_tensor = Tensor::new(DataType::Float, shape.to_vec());
        let mut out_tensor = Tensor::new(DataType::Float, shape.to_vec());

        let lhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
        let rhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();

        lhs_tensor.init(lhs_data);
        rhs_tensor.init(rhs_data);

        cuda._matmul(&lhs_tensor, &rhs_tensor, &mut out_tensor)
            .unwrap();
        let result = cuda.dtoh_f32(out_tensor.data().unwrap()).unwrap();
        assert_eq!(result, vec![7.0, 10.0, 15.0, 22.0])
    }
}
