use crate::error::{Error, Result};
use crate::kernels::cast::CastKernel;
use crate::kernels::{add, cast, div, mul, reduce_mean, sqrt, whereop};
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

pub fn load_add_kernel_alpha_beta_inplace(
    device: &Arc<CudaDevice>,
    dtype: DataType,
) -> Result<CudaFunction> {
    let (fwd_fn_name, fwd_fn_all, module_name, ptx_src) = (
        add::FWD_FN_NAMES_ALPHA_BETA_INPLACE[dtype as usize],
        add::FWD_FN_NAMES_ALPHA_BETA_INPLACE.as_slice(),
        add::MODULE_NAME,
        add::PTX_SRC,
    );

    if !device.has_func(module_name, fwd_fn_name) {
        device
            .load_ptx(ptx_src.into(), module_name, fwd_fn_all)
            .map_err(|e| Error::Internal(format!("failed to load kernel: {e:?}")))?
    }

    // Todo: circle back and assess if it's safe to unwrap.
    Ok(device
        .get_func(module_name, fwd_fn_name)
        .expect("To have been loaded"))
}

pub fn load_kernel(device: &Arc<CudaDevice>, op: Op, dtype: DataType) -> Result<CudaFunction> {
    let (fwd_fn_name, fwd_fn_all, module_name, ptx_src) = match op {
        Op::Add => (
            add::FWD_FN_NAMES[dtype as usize],
            add::FWD_FN_NAMES.as_slice(),
            add::MODULE_NAME,
            add::PTX_SRC,
        ),
        Op::Div => (
            div::FWD_FN_NAMES[dtype as usize],
            div::FWD_FN_NAMES.as_slice(),
            div::MODULE_NAME,
            div::PTX_SRC,
        ),
        Op::Mul => (
            mul::FWD_FN_NAMES[dtype as usize],
            mul::FWD_FN_NAMES.as_slice(),
            mul::MODULE_NAME,
            mul::PTX_SRC,
        ),
        Op::ReduceMean => (
            reduce_mean::FWD_FN_NAMES[dtype as usize],
            reduce_mean::FWD_FN_NAMES,
            reduce_mean::MODULE_NAME,
            reduce_mean::PTX_SRC,
        ),
        Op::Sqrt => (
            sqrt::FWD_FN_NAMES[dtype as usize],
            sqrt::FWD_FN_NAMES.as_slice(),
            sqrt::MODULE_NAME,
            sqrt::PTX_SRC,
        ),
        Op::Where => (
            whereop::FWD_FN_NAMES[dtype as usize],
            whereop::FWD_FN_NAMES.as_slice(),
            whereop::MODULE_NAME,
            whereop::PTX_SRC,
        ),
        _ => unimplemented!(),
    };

    if !device.has_func(module_name, fwd_fn_name) {
        device
            .load_ptx(ptx_src.into(), module_name, fwd_fn_all)
            .map_err(|e| Error::Internal(format!("failed to load kernel: {e:?}")))?
    }

    // Todo: circle back and assess if it's safe to unwrap.
    Ok(device
        .get_func(module_name, fwd_fn_name)
        .expect("To have been loaded"))
}

pub fn load_cast_kernel(device: &Arc<CudaDevice>, kernel_name: CastKernel) -> Result<CudaFunction> {
    if !device.has_func(cast::MODULE_NAME, kernel_name.as_str()) {
        device
            .load_ptx(cast::PTX_SRC.into(), cast::MODULE_NAME, cast::FWD_FN_NAMES)
            .map_err(|e| Error::Internal(format!("failed to load kernel: {e:?}")))?
    }

    // Todo: circle back and assess if it's safe to unwrap.
    Ok(device
        .get_func(cast::MODULE_NAME, kernel_name.as_str())
        .expect("To have been loaded"))
}
