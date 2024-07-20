use crate::Result;
use crate::{Context, Error};

pub fn compute<T>(ctx: &mut Context<T>) -> Result<()> {
    let x = ctx.get_input(0)?;

    if x.shape().len() == 0 {
        return Err(Error::InvalidTensorDimensions);
    }

    let mut y_shape = Box::new([0; 2]);
    let axis = ctx
        .get_attributes()
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
            return Err(Error::InvalidAttribute);
        }
        axis => {
            let axis = axis.unsigned_abs() as usize;
            y_shape[0] = x.shape()[..axis].iter().product();
            y_shape[1] = x.shape()[axis..].iter().product();
        }
    }

    let y = ctx.get_output_mut(0)?;
    y.reshape(y_shape.to_vec());
    // Todo: This tensor needs to point to data in input, x.
    // This way we can avoid making a copy of the same data.
    Ok(())
}
