use crate::device::cuda::data::Data;
use crate::device::cuda::kernels::{add, mul};
use crate::device::Result;
use crate::device::{cuda, Device, Error};
use crate::dtype::DType;
use crate::op::Op;
use crate::tensor::Tensor;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use half::f16;
use std::alloc::Allocator;
use std::sync::Arc;

pub struct Cuda<A> {
    device: Arc<CudaDevice>,
    alloc: A,
}

impl<A> Clone for Cuda<A>
where
    A: Allocator + Clone,
{
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            alloc: self.alloc.clone(),
        }
    }
}

impl<A> Cuda<A>
where
    A: Allocator + Clone,
{
    pub fn new(device: Arc<CudaDevice>, alloc: A) -> Self {
        Self { alloc, device }
    }

    pub fn matmul(
        &self,
        lhs: &Tensor<Self>,
        rhs: &Tensor<Self>,
        out: &mut Tensor<Self>,
    ) -> Result<()> {
        let lhs_shape = lhs.shape();
        let b = lhs_shape[..lhs_shape.len() - 2].iter().product::<usize>();
        let m = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];

        let rhs_shape = rhs.shape();
        let n = rhs_shape[rhs_shape.len() - 2];

        match *lhs.dtype() {
            DType::F16 => {
                todo!()
            }
            DType::F32 => {
                let lhs_stride = lhs.stride();
                let rhs_stride = rhs.stride();
                let config = cuda::ops::matmul::gemm_config::<f32>(
                    1.0,
                    0.0,
                    (b, m, n, k),
                    (lhs_shape, lhs_stride),
                    (rhs_shape, rhs_stride),
                )
                .unwrap();
                let mut out = unsafe { self.device.alloc::<f32>(b * m * n).unwrap() };
                let cublas = CudaBlas::new(self.device.clone()).unwrap();
                unsafe {
                    cuda::ops::matmul::gemm_stride_batched_f32(
                        &cublas,
                        config,
                        &rhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                        &lhs.data().ok_or(()).unwrap().f32().unwrap().slice(..),
                        &mut out,
                    )
                    .unwrap();
                }
            }
            DType::F64 => {
                todo!()
            }
        }
        Ok(())
    }

    fn kernel(&self, op: Op, dtype: DType) -> cuda::Result<CudaFunction> {
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
        };

        if !self.device.has_func(module_name, fwd_fn_name) {
            self.device
                .load_ptx(ptx_src.into(), module_name, fwd_fn_all)
                .map_err(|_| ())?
        }
        Ok(self
            .device
            .get_func(module_name, fwd_fn_name)
            .expect("To have been loaded"))
    }

    pub fn forward(
        &self,
        op: Op,
        lhs: &Tensor<Self>,
        rhs: &Tensor<Self>,
        out: &mut Tensor<Self>,
    ) -> cuda::Result<()> {
        // Todo: more assertions here.
        debug_assert!(lhs.shape() == rhs.shape());

        // Todo: Validate that the tensors are valid for the operation.

        let func = self.kernel(op, *lhs.dtype())?;

        // Todo: should we directly initialize this in the device?
        let mut info: Vec<usize, A> =
            Vec::with_capacity_in(3 * lhs.shape().len(), self.alloc.clone());
        info.extend(lhs.shape());
        info.extend(lhs.stride());
        info.extend(rhs.stride());
        let info = self
            .device
            .htod_copy(info.as_slice().to_vec())
            .map_err(|_| ())?;

        let elem_count: usize = lhs.shape().iter().product();

        let num_threads = 128;
        let num_blocks = (elem_count + num_threads - 1) / num_threads;
        let config = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (num_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if matches!(lhs.dtype(), &DType::F16) {
            let lhs_data = lhs.data().ok_or(())?.f16()?;
            let rhs_data = rhs.data().ok_or(())?.f16()?;
            let mut out_slice = unsafe { self.device.alloc::<f16>(elem_count).unwrap() };

            let params = (
                elem_count,
                lhs.shape().len(),
                &info,
                lhs_data,
                rhs_data,
                &mut out_slice,
            );
            unsafe { func.launch(config, params).map_err(|_| ())? };
            let _ = out.set_data(Arc::new(Data::F16(out_slice)));
        } else if matches!(lhs.dtype(), &DType::F32) {
            let lhs_data = lhs.data().ok_or(())?.f32()?;
            let rhs_data = rhs.data().ok_or(())?.f32()?;
            let mut out_slice = unsafe { self.device.alloc::<f32>(elem_count).unwrap() };

            let params = (
                elem_count,
                lhs.shape().len(),
                &info,
                lhs_data,
                rhs_data,
                &mut out_slice,
            );
            unsafe { func.launch(config, params).map_err(|_| ())? };
            let _ = out.set_data(Arc::new(Data::F32(out_slice)));
        } else if matches!(lhs.dtype(), &DType::F64) {
            let lhs_data = rhs.data().ok_or(())?.f64()?;
            let rhs_data = lhs.data().ok_or(())?.f64()?;
            let mut out_slice = unsafe { self.device.alloc::<f64>(elem_count).unwrap() };

            let params = (
                elem_count,
                lhs.shape().len(),
                &info,
                lhs_data,
                rhs_data,
                &mut out_slice,
            );
            unsafe { func.launch(config, params).map_err(|_| ())? };
            let _ = out.set_data(Arc::new(Data::F64(out_slice)));
        }

        Ok(())
    }
}

impl<A> Device for Cuda<A>
where
    A: Allocator + Clone,
{
    type Data = Data;
    type Cpu = A;

    fn htod_f16(&self, data: Vec<f16>) -> Result<Self::Data> {
        self.device
            .htod_copy(data)
            .map(Data::F16)
            .map_err(|_| Error::Unknown)
    }

    fn htod_f32(&self, data: Vec<f32>) -> Result<Self::Data> {
        self.device
            .htod_copy(data)
            .map(Data::F32)
            .map_err(|_| Error::Unknown)
    }

    fn htod_f64(&self, data: Vec<f64>) -> Result<Self::Data> {
        self.device
            .htod_copy(data)
            .map(Data::F64)
            .map_err(|_| Error::Unknown)
    }

    fn dtoh_f16(&self, data: &Self::Data) -> Result<Vec<f16>> {
        let data = data.f16().map_err(|_| Error::Unknown)?;
        self.device.dtoh_sync_copy(data).map_err(|_| Error::Unknown)
    }

    fn dtoh_f32(&self, data: &Self::Data) -> Result<Vec<f32>> {
        let data = data.f32().map_err(|_| Error::Unknown)?;
        self.device.dtoh_sync_copy(data).map_err(|_| Error::Unknown)
    }

    fn dtoh_f64(&self, data: &Self::Data) -> Result<Vec<f64>> {
        let data = data.f64().map_err(|_| Error::Unknown)?;
        self.device.dtoh_sync_copy(data).map_err(|_| Error::Unknown)
    }
}

#[cfg(test)]
mod test {
    use crate::device::cuda::cuda::Cuda;
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::op::Op;
    use crate::tensor::Tensor;
    use cudarc::driver::CudaDevice;
    use half::f16;
    use std::alloc::Global;
    use std::sync::Arc;

    #[test]
    fn test_add_f16() {
        let device = CudaDevice::new(0).unwrap();
        let cuda = Cuda::new(device, Global);

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

        let mut lhs_tensor = Tensor::new(DType::F16, shape.to_vec(), lhs_strides.to_vec());
        let mut out_tensor = Tensor::new(DType::F16, shape.to_vec(), lhs_strides.to_vec());
        let mut rhs_tensor = Tensor::new(DType::F16, shape.to_vec(), rhs_strides.to_vec());

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

        lhs_tensor.set_data(Arc::new(lhs_data));
        rhs_tensor.set_data(Arc::new(rhs_data));

        cuda.forward(Op::Add, &lhs_tensor, &rhs_tensor, &mut out_tensor)
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
        let cuda = Cuda::new(device, Global);

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

        let mut lhs_tensor = Tensor::new(DType::F32, shape.to_vec(), lhs_strides.to_vec());
        let mut rhs_tensor = Tensor::new(DType::F32, shape.to_vec(), rhs_strides.to_vec());
        let mut out_tensor = Tensor::new(DType::F32, shape.to_vec(), lhs_strides.to_vec());

        let lhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();
        let rhs_data = cuda.htod_f32(vec![1f32, 2f32, 3f32, 4f32]).unwrap();

        lhs_tensor.set_data(Arc::new(lhs_data));
        rhs_tensor.set_data(Arc::new(rhs_data));

        cuda.forward(Op::Add, &lhs_tensor, &rhs_tensor, &mut out_tensor)
            .unwrap();
        let result = cuda.dtoh_f32(out_tensor.data().unwrap()).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }

    #[test]
    fn test_mul_f16() {
        let device = CudaDevice::new(0).unwrap();
        let cuda = Cuda::new(device, Global);

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

        let mut lhs_tensor = Tensor::new(DType::F16, shape.to_vec(), lhs_strides.clone());
        let mut rhs_tensor = Tensor::new(DType::F16, shape.to_vec(), rhs_strides);
        let mut out_tensor = Tensor::new(DType::F16, shape.to_vec(), lhs_strides);

        let lhs_data = cuda.htod_f16(vec![f16::from_f32(2.0); elem_num]).unwrap();
        let rhs_data = cuda.htod_f16(vec![f16::from_f32(3.0); elem_num]).unwrap();

        lhs_tensor.set_data(Arc::new(lhs_data));
        rhs_tensor.set_data(Arc::new(rhs_data));

        cuda.forward(Op::Mul, &lhs_tensor, &rhs_tensor, &mut out_tensor)
            .unwrap();
        let result = cuda.dtoh_f16(out_tensor.data().unwrap()).unwrap();
        assert_eq!(result, vec![f16::from_f32(6.0); elem_num])
    }
}
