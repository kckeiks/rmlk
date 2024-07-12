use crate::cuda::data::CudaData;
use crate::cuda::kernels::{add, mul};
use crate::cuda::ops;
use crate::kernel::Context;
use crate::kernel::Kernel;
use crate::{Error, Result};
use cudarc::driver::{CudaDevice, CudaFunction};
use rmlk_ir::{DataType, Op};
use std::sync::Arc;

pub struct CudaKernel {
    op: Op,
    device: Arc<CudaDevice>,
}

impl CudaKernel {
    pub fn new(op: Op, device: Arc<CudaDevice>) -> Self {
        Self { op, device }
    }

    fn kernel(&self, dtype: DataType) -> Result<CudaFunction> {
        let (fwd_fn_name, fwd_fn_all, module_name, ptx_src) = match self.op {
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
}

impl Kernel for CudaKernel {
    type Data = CudaData;

    fn compute(&self, ctx: &mut Context<Self::Data>) -> Result<()> {
        match self.op {
            Op::Gemm => {
                ops::gemm::compute(ctx, self.device.clone())?;
            }
            Op::MatMul => {
                ops::gemm::compute(ctx, self.device.clone())?;
            }
            Op::Add => {
                // Todo: More validation.
                let dtype = ctx.get_input(0)?.dtype();
                let func = self.kernel(*dtype)?;
                ops::add::compute(ctx, self.device.clone(), func)?;
            }
            Op::Conv => {
                ops::conv::compute(ctx, self.device.clone())?;
            }
            _ => todo!(),
        }

        Ok(())
    }
}
