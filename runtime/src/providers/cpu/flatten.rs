use crate::core::error::{InternalError, Result};
use crate::core::{device_service::DeviceService, Context};

pub struct FlattenTemplate(());

impl FlattenTemplate {
    pub fn new() -> Self {
        Self(())
    }
    pub fn compute<T: DeviceService>(self, ctx: &mut Context<T>) -> Result<()> {
        _compute(ctx)
    }
}

pub fn _compute<T: DeviceService>(ctx: &mut Context<T>) -> Result<()> {
    let x = ctx.get_input(0)?;

    if x.shape().len() == 0 {
        return Err(InternalError::InvalidTensorShape {
            shape: x.shape().to_vec(),
        });
    }

    let mut y_shape = [0; 2];
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
            return Err(InternalError::InvalidAxis { axis });
        }
        axis => {
            let axis = axis.unsigned_abs() as usize;
            y_shape[0] = x.shape()[..axis].iter().product();
            y_shape[1] = x.shape()[axis..].iter().product();
        }
    }

    // Todo: This tensor needs to point to data in input, x.
    // This way we can avoid making a copy of the same data.
    let dtype = *x.dtype();
    let dev_data = x.dev_data_view().ok_or(InternalError::MissingDeviceData)?;
    let shape = ctx
        .execution_state()
        .scratch_alloc()
        .allocate_from_slice(y_shape.as_slice())?;
    let mut y = ctx.get_output_mut(0)?;
    y.reshape(shape)?;
    y.set_dev_data_from_view(dev_data);
    y.set_dtype(dtype);

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::core::Context;
    use crate::ops::flatten::_compute;
    use crate::test_utils;
    use crate::test_utils::{MockProvider, TestNode, TestParams};
    use rmlk_schema::{DataType, Op};

    #[test]
    fn test_flatten_f32() {
        let shape = vec![1, 1, 4, 4];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        };

        let params = TestParams {
            inputs: vec![node_a],
            attributes: Vec::new(),
            op: Op::Flatten,
        };

        let mut state = test_utils::build_graph_and_state(MockProvider::new(), params);
        let mut context = Context::new(&mut state, 2).unwrap();

        _compute(&mut context).unwrap();

        let shape = context.get_output(0).unwrap().shape();

        assert_eq!(shape, &vec![1, 16])
    }
}
