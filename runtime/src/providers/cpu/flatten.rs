use crate::core::error::InternalError;
use crate::core::{device_service::DeviceService, Context};
use anyhow::Result;
use log::debug;
use std::fmt::{Display, Formatter};

pub struct FlattenTemplate(());

impl FlattenTemplate {
    pub fn new() -> Self {
        Self(())
    }
    pub fn compute<T: DeviceService>(self, ctx: &mut Context<T>) -> Result<()> {
        compute(ctx)
    }
}

pub fn compute<T: DeviceService>(ctx: &mut Context<T>) -> Result<()> {
    let x = ctx.get_input(0)?;

    debug!("[x][shape={:?}][stride=[{:?}]", x.shape(), x.stride());

    if x.shape().len() == 0 {
        return Err(FlattenError::InvalidInputShape.into());
    }

    let mut y_shape = [0; 2];
    let axis = ctx
        .get_attributes()
        .as_ref()
        .map(|attrs| attrs.get("axis"))
        .flatten()
        .and_then(|attr| attr.int())
        .unwrap_or(1);
    match axis {
        0 => {
            y_shape[0] = 1;
            y_shape[1] = x.shape().iter().product();
        }
        axis if axis.unsigned_abs() as usize >= x.shape().len() => {
            return Err(FlattenError::AxisOutOfShape.into());
        }
        axis => {
            let axis = axis.unsigned_abs() as usize;
            y_shape[0] = x.shape()[..axis].iter().product();
            y_shape[1] = x.shape()[axis..].iter().product();
        }
    }

    let dev_data = x
        .dev_data_ptr_clone()
        .ok_or(InternalError::MissingDeviceData)?;
    let mut y = ctx.get_output(0)?;
    y.set_dev_data_ptr(dev_data);

    let index = y.dst_id();
    ctx.execution_state_mut()
        .copy_shape_from_slice(&y_shape, index)?;

    let y = ctx.get_output(0)?;
    debug!("[y][shape={:?}][stride=[{:?}]", y.shape(), y.stride());

    Ok(())
}

#[derive(Debug)]
pub enum FlattenError {
    AxisOutOfShape,
    InvalidInputShape,
}

impl Display for FlattenError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for FlattenError {}
