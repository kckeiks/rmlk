use crate::core::error::InternalError;
use crate::core::Context;
use crate::providers::cuda::Cuda;
use anyhow::Result;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use half::f16;
use log::debug;
use num_traits::Num;
use rmlk_schema::{DataType, DataTypeMap};
use std::fmt::{Display, Formatter};

// Todo: this should be generic.
pub struct FlattenTemplate(());

impl FlattenTemplate {
    pub fn new() -> Self {
        Self(())
    }
    pub fn compute(self, ctx: &mut Context<Cuda>) -> Result<()> {
        let dtype = ctx.get_input(0)?.dtype();

        match dtype {
            DataType::Float16 => compute::<f16>(ctx),
            DataType::Float => compute::<f32>(ctx),
            DataType::Double => compute::<f64>(ctx),
            DataType::Int32 => compute::<i32>(ctx),
            DataType::Uint32 => compute::<u32>(ctx),
            DataType::Int64 => compute::<i64>(ctx),
            DataType::Uint64 => compute::<u64>(ctx),
            _ => Err(InternalError::UnsupportedDataType { dtype }.into()),
        }
    }
}

pub fn compute<T>(ctx: &mut Context<Cuda>) -> Result<()>
where
    T: DataTypeMap + ValidAsZeroBits + DeviceRepr + Num,
{
    let x = ctx.get_input(0)?;

    debug!(
        "[x][dtype={:?}][shape={:?}][stride=[{:?}]",
        x.dtype(),
        x.shape(),
        x.stride()
    );

    let mut y_shape = [0; 2];
    let axis = ctx
        .get_attributes()
        .as_ref()
        .map(|attrs| attrs.get("axis"))
        .flatten()
        .and_then(|attr| attr.int())
        .unwrap_or(1);
    match axis {
        0 if x.is_scalar() => {
            y_shape[0] = 1;
            y_shape[1] = 1;
        }
        0 => {
            y_shape[0] = 1;
            y_shape[1] = x.shape().iter().product();
        }
        _axis if x.is_scalar() => {
            return Err(FlattenError::InvalidInputShape.into());
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

    let y = ctx.get_output(0)?;
    y.copy_shape_from_slice(&y_shape);

    let x_payload = x.payload();
    let x_data = x_payload.data::<T>();
    y.write_payload(&x_data)?;

    debug!(
        "[y][dtype={:?}][shape={:?}][stride=[{:?}]",
        y.dtype(),
        y.shape(),
        y.stride()
    );

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
