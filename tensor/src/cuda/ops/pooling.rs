use crate::cuda::data::CudaData;
use crate::{Context, Error, Result};
use cudarc::cudnn::{Cudnn};
use cudarc::driver::{CudaDevice};

use std::sync::Arc;

pub fn compute(ctx: &mut Context<CudaData>, device: Arc<CudaDevice>) -> Result<()> {
    let cudnn = Cudnn::new(device.clone()).map_err(|_| Error::CudnnInternal)?;
    // Input data tensor.
    let x = ctx.get_input(0)?;
    // Todo: Let's define an attributes object.


    // Todo: Finish.
    Ok(())
}
