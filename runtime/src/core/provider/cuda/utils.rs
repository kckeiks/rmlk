use crate::core::error::{Error, Result};
use cudarc::driver::{CudaDevice, CudaFunction};
use rmlk_ir::{DataType, Op};
use rmlk_tensor::cuda::{add, mul};
use std::sync::Arc;

pub fn load_kernel(device: Arc<CudaDevice>, op: Op, dtype: DataType) -> Result<CudaFunction> {
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

    if !device.has_func(module_name, fwd_fn_name) {
        device
            .load_ptx(ptx_src.into(), module_name, fwd_fn_all)
            .map_err(|_| Error::FailedToLoadKernel)?
    }

    Ok(device
        .get_func(module_name, fwd_fn_name)
        .expect("To have been loaded"))
}
