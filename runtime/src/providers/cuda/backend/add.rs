use crate::core::error::{InternalError, Result};
use crate::core::Context;
use crate::providers::cuda::data::CudaData;
use crate::providers::cuda::Cuda;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DeviceSlice, ValidAsZeroBits,
};
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap, Op};
use std::sync::Arc;

pub struct AdditionBackend {
    device: Arc<CudaDevice>,
    f: CudaFunction,
}

impl AdditionBackend {
    pub fn new(device: Arc<CudaDevice>, f: CudaFunction) -> Self {
        Self { device, f }
    }
}

impl AdditionBackend {
    fn compute_addition<D, T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        D: DataTypeMap + CudnnDataType + ValidAsZeroBits + DeviceRepr + Num,
        T: AdditionKernel,
    {
        // The output should have the same dimensions.
        // We do it now to avoid lifetime errors.
        {
            let a = ctx.get_input(0)?;
            let c = ctx.get_output(0)?;
            let a_index = a.src_id();
            let c_index = c.dst_id();
            ctx.execution_state_mut()
                .copy_shape_from_within(a_index, c_index)?;
        }

        let a = ctx.get_input(0)?;
        let b = ctx.get_input(1)?;

        #[cfg(debug_assertions)]
        {
            let a = ctx.get_input(0)?;
            let b = ctx.get_input(1)?;
            let c = ctx.get_output(0)?;
            debug!("[a][add][shape={:?}][stride=[{:?}]", a.shape(), a.stride());
            debug!("[b][add][shape={:?}][stride=[{:?}]", b.shape(), b.stride());
            debug!("[c][add][shape={:?}][stride=[{:?}]", c.shape(), c.stride());
        }

        let scratch_alloc = ctx.execution_state().scratch_alloc();
        let info_buffer = scratch_alloc.allocate(3 * a.shape().len())?;

        let a_dev_data_ref = a.try_dev_data_ptr()?;
        let a_dev_data = a_dev_data_ref.data::<D>();

        let b_dev_data_ref = b.try_dev_data_ptr()?;
        let b_dev_data = b_dev_data_ref.data::<D>();

        let elem_count: usize = a.shape().iter().product();

        // Allocate device data for the tensor if we haven't done it yet
        // or if the existing allocated data has a different size.
        {
            let mut c = ctx.get_output(0)?;
            let c_dev_data_ref = c.dev_data_ptr_mut();
            let need_to_alloc_dev_data = c_dev_data_ref.is_none()
                || c_dev_data_ref
                    .as_ref()
                    .map(|data| data.data::<D>().len() != elem_count)
                    .unwrap_or(true);

            // We need to remove this immutable reference so we can mutate `y`.
            drop(c_dev_data_ref);

            if need_to_alloc_dev_data {
                let c_dev_data = self
                    .device
                    .alloc_zeros::<f32>(a.shape().iter().copied().product::<usize>())
                    .map_err(rmlk_cuda::Error::from)?;
                c.set_dev_data(CudaData::new(c_dev_data));
            };
        }

        // The device data should exist so we will execute the kernel
        // and update the destination device data with the result.
        let c = ctx.get_output(0)?;
        let mut c_dev_data_ref = c.dev_data_ptr_mut();
        let mut c_dev_data = c_dev_data_ref
            .as_mut()
            .expect("we already checked that it initialized")
            .data_mut();

        T::execute::<D>(
            self.device,
            self.f,
            &a_dev_data,
            &a.shape(),
            &a.stride(),
            &b_dev_data,
            &b.shape(),
            &b.stride(),
            &mut c_dev_data,
            info_buffer,
        )?;

        Ok(())
    }

    pub fn compute<T>(self, ctx: &mut Context<Cuda>) -> Result<()>
    where
        T: AdditionKernel,
    {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float => self.compute_addition::<f32, T>(ctx),
            _ => Err(InternalError::UnsupportedOpForDataType { op: Op::Add, dtype }),
        }
    }
}

pub trait AdditionKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        a_dev_data: &CudaSlice<T>,
        a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        b_shape: &[usize],
        b_stride: &[usize],
        c_dev_data: &mut CudaSlice<T>,
        info_buffer: &mut [usize],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr;
}

pub struct ActiveKernel(());

impl AdditionKernel for ActiveKernel {
    fn execute<T>(
        device: Arc<CudaDevice>,
        func: CudaFunction,
        a_dev_data: &CudaSlice<T>,
        a_shape: &[usize],
        a_stride: &[usize],
        b_dev_data: &CudaSlice<T>,
        b_shape: &[usize],
        b_stride: &[usize],
        y_dev_data: &mut CudaSlice<T>,
        info_buffer: &mut [usize],
    ) -> Result<()>
    where
        T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
    {
        rmlk_cuda::kernels::add::compute(
            device,
            func,
            a_dev_data,
            a_shape,
            a_stride,
            b_dev_data,
            b_shape,
            b_stride,
            y_dev_data,
            info_buffer,
        )
        .map_err(Into::into)
    }
}

pub struct NoOpKernel(());

impl AdditionKernel for NoOpKernel {
    fn execute<T>(
        _: Arc<CudaDevice>,
        _: CudaFunction,
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &CudaSlice<T>,
        _: &[usize],
        _: &[usize],
        _: &mut CudaSlice<T>,
        _: &mut [usize],
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::providers::cuda::data::CudaData;
    use crate::providers::cuda::kernel::add::BackendHandler;
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
        let cuda_kernel = BackendHandler::new(device.clone(), f);
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .dev_data_ptr()
            .unwrap()
            .f32()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }
}
