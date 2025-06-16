use crate::error::Result;
use crate::kernels::cast::CastKernel;
use crate::kernels::{add, cast, expand, mul, reduce_mean, sqrt, whereop};
use cudarc::driver::{CudaContext, CudaFunction};
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
    ctx: &Arc<CudaContext>,
    dtype: DataType,
) -> Result<CudaFunction> {
    let (fwd_fn_name, _, _, ptx_src) = (
        add::FWD_FN_NAMES_ALPHA_BETA_INPLACE[dtype as usize],
        add::FWD_FN_NAMES_ALPHA_BETA_INPLACE.as_slice(),
        add::MODULE_NAME,
        add::PTX_SRC,
    );

    let module = ctx.load_module(ptx_src.into())?;
    module.load_function(fwd_fn_name).map_err(Into::into)
}

// Todo: Clean up.
pub fn load_kernel(ctx: &Arc<CudaContext>, op: Op, dtype: DataType) -> Result<CudaFunction> {
    let (fwd_fn_name, _, _, ptx_src) = match op {
        Op::Add => (
            add::FWD_FN_NAMES[dtype as usize],
            add::FWD_FN_NAMES.as_slice(),
            add::MODULE_NAME,
            add::PTX_SRC,
        ),
        Op::Expand => (
            expand::FWD_FN_NAMES[dtype as usize],
            expand::FWD_FN_NAMES,
            expand::MODULE_NAME,
            expand::PTX_SRC,
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

    let module = ctx.load_module(ptx_src.into())?;
    module.load_function(fwd_fn_name).map_err(Into::into)
}

pub fn load_kernel_v2(
    ctx: &Arc<CudaContext>,
    ptx_src: &str,
    fn_name: &str,
) -> Result<CudaFunction> {
    let module = ctx.load_module(ptx_src.into())?;
    module.load_function(fn_name).map_err(Into::into)
}

pub fn load_cast_kernel(ctx: &Arc<CudaContext>, kernel_name: CastKernel) -> Result<CudaFunction> {
    let module = ctx.load_module(cast::PTX_SRC.into())?;
    module
        .load_function(kernel_name.as_str())
        .map_err(Into::into)
}
