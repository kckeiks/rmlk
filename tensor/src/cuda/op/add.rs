use crate::cuda::data::CudaData;
use crate::kernel::Context;
use crate::Error;
use crate::Result;
use cudarc::cudnn::CudnnDataType;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use half::f16;
use rmlk_ir::DataType;
use std::sync::Arc;

pub fn compute_v2<T>(
    device: Arc<CudaDevice>,
    func: CudaFunction,
    lhs_data: &CudaSlice<T>,
    lhs_shape: &[usize],
    lhs_stride: &[usize],
    rhs_data: &CudaSlice<T>,
    rhs_shape: &[usize],
    rhs_stride: &[usize],
) -> Result<CudaSlice<T>>
where
    T: CudnnDataType + ValidAsZeroBits + DeviceRepr,
{
    // Todo: more assertions here.
    debug_assert!(lhs_shape == rhs_shape);

    // Todo: Validate that the tensors are valid for the operation.
    // Todo: should we directly initialize this in the device?
    let mut info: Vec<usize> = Vec::with_capacity(3 * lhs_shape.len());
    info.extend(lhs_shape);
    info.extend(lhs_stride);
    info.extend(rhs_stride);

    let info = device
        .htod_copy(info.as_slice().to_vec())
        .map_err(|_| Error::Unknown)?;

    let elem_count: usize = lhs_shape.iter().product();
    let num_threads = 128;
    let num_blocks = (elem_count + num_threads - 1) / num_threads;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut out_slice = unsafe { device.alloc::<T>(elem_count).unwrap() };

    let params = (
        elem_count,
        lhs_shape.len(),
        &info,
        lhs_data,
        rhs_data,
        &mut out_slice,
    );

    unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };

    return Ok(out_slice);
}

pub fn compute(
    ctx: &mut Context<CudaData>,
    device: Arc<CudaDevice>,
    func: CudaFunction,
) -> Result<()> {
    let lhs = ctx.get_input(0)?;
    let rhs = ctx.get_input(1)?;

    // Todo: more assertions here.
    debug_assert!(lhs.shape() == rhs.shape());

    // Todo: Validate that the tensors are valid for the operation.
    // Todo: should we directly initialize this in the device?
    let mut info: Vec<usize> = Vec::with_capacity(3 * lhs.shape().len());
    info.extend(lhs.shape());
    info.extend(lhs.stride());
    info.extend(rhs.stride());
    let info = device
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

    let lhs_shape_size = lhs.shape().len();

    if matches!(lhs.dtype(), &DataType::Float16) {
        let lhs_data = lhs.data().ok_or(Error::Executor)?.f16()?;
        let rhs_data = rhs.data().ok_or(Error::Executor)?.f16()?;

        let mut out_slice = unsafe { device.alloc::<f16>(elem_count).unwrap() };

        let params = (
            elem_count,
            lhs_shape_size,
            &info,
            lhs_data,
            rhs_data,
            &mut out_slice,
        );
        unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };
        let out = ctx.get_output_mut(0)?;
        let _ = out.init(CudaData::F16(out_slice));
    } else if matches!(lhs.dtype(), &DataType::Float) {
        let lhs_data = lhs.data().ok_or(Error::Executor)?.f32()?;
        let rhs_data = rhs.data().ok_or(Error::Executor)?.f32()?;

        let mut out_slice = unsafe { device.alloc::<f32>(elem_count).unwrap() };

        let params = (
            elem_count,
            lhs.shape().len(),
            &info,
            lhs_data,
            rhs_data,
            &mut out_slice,
        );
        unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };
        let out = ctx.get_output_mut(0)?;
        let _ = out.init(CudaData::F32(out_slice));
    } else if matches!(lhs.dtype(), &DataType::Double) {
        let lhs_data = rhs.data().ok_or(Error::Executor)?.f64()?;
        let rhs_data = lhs.data().ok_or(Error::Executor)?.f64()?;

        let mut out_slice = unsafe { device.alloc::<f64>(elem_count).unwrap() };

        let params = (
            elem_count,
            lhs.shape().len(),
            &info,
            lhs_data,
            rhs_data,
            &mut out_slice,
        );
        unsafe { func.launch(config, params).map_err(|_| Error::Executor)? };
        let out = ctx.get_output_mut(0)?;
        let _ = out.init(CudaData::F64(out_slice));
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::cuda::data::CudaData;
    use crate::cuda::kernel::CudaKernel;
    use crate::cuda::op::add::compute_v2;
    use crate::kernel::{Context, Kernel};
    use crate::test_utils::{TestNode, TestParams};
    use crate::{test_utils, Tensor};
    use cudarc::driver::CudaDevice;
    use half::f16;
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_add_f32_v2() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![4, 1, 1, 1];

        let lhs = Tensor::<CudaData>::new_with_shape(DataType::Float, shape.clone());
        let lhs_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let rhs = Tensor::<CudaData>::new_with_shape(DataType::Float, shape);
        let rhs_data = device.htod_copy(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let cuda_wrapper = CudaKernel::new(Op::Add, device.clone());
        let func = cuda_wrapper.kernel(DataType::Float).unwrap();

        let out_data = compute_v2::<f32>(
            device.clone(),
            func,
            &lhs_data,
            lhs.shape(),
            lhs.stride(),
            &rhs_data,
            rhs.shape(),
            rhs.stride(),
        )
        .unwrap();
        let result = device.dtoh_sync_copy(&out_data).unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0])
    }

    #[test]
    fn test_add_f16() {
        let device = CudaDevice::new(0).unwrap();
        let shape = vec![4, 1, 1, 1];
        let dtype = DataType::Float16;

        let node_a = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F16(
                device
                    .htod_copy(vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ])
                    .unwrap(),
            )),
        };
        let node_b = TestNode {
            shape: shape.clone(),
            dtype,
            data: Some(CudaData::F16(
                device
                    .htod_copy(vec![
                        f16::from_f32(1.0),
                        f16::from_f32(2.0),
                        f16::from_f32(3.0),
                        f16::from_f32(4.0),
                    ])
                    .unwrap(),
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

        let (_, mut state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Add, device.clone());
        cuda_kernel.compute(&mut context).unwrap();

        let out_data = context
            .get_output(0)
            .unwrap()
            .data()
            .unwrap()
            .f16()
            .unwrap();
        let result = device.dtoh_sync_copy(out_data).unwrap();

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

        let (_, mut state) = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 2).unwrap();

        let cuda_kernel = CudaKernel::new(Op::Add, device.clone());
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
