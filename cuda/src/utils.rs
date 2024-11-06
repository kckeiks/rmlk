use crate::error::{Error, Result};
use crate::kernels::{add, mul};
use cudarc::driver::{CudaDevice, CudaFunction};
#[cfg(test)]
use num_traits::Num;
use rmlk_schema::{DataType, Op};
#[cfg(test)]
use std::ops::AddAssign;
use std::sync::Arc;

#[cfg(test)]
pub fn calculate_stride<T: Num + Copy + AddAssign>(shape: &[T], stride: &mut [T]) {
    let dims = shape.len();

    debug_assert_eq!(dims, stride.len());

    stride[dims - 1] = T::one();
    for i in (0..dims - 1).rev() {
        stride[i] += stride[i + 1] * shape[i + 1];
    }
}

pub fn load_kernel(device: &Arc<CudaDevice>, op: Op, dtype: DataType) -> Result<CudaFunction> {
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
            .map_err(|e| Error::Internal(format!("failed to load kernel: {e:?}")))?
    }

    Ok(device
        .get_func(module_name, fwd_fn_name)
        .expect("To have been loaded"))
}
