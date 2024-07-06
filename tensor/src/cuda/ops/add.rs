use crate::cuda::data::CudaData;
use crate::kernel::Context;
use crate::Error;
use crate::Result;
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use half::f16;
use rmlk_ir::DataType;
use std::sync::Arc;

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
