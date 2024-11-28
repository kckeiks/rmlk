use crate::core::kernel::{KernelError, Result};
use crate::core::{device_service::DeviceService, Context};

pub struct FlattenOp(());

impl FlattenOp {
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
        return Err(KernelError::InvalidTensorDimensions(x.shape().to_vec()));
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
            return Err(KernelError::Other(format!("invalid axis `{axis}`")));
        }
        axis => {
            let axis = axis.unsigned_abs() as usize;
            y_shape[0] = x.shape()[..axis].iter().product();
            y_shape[1] = x.shape()[axis..].iter().product();
        }
    }

    // Todo: This tensor needs to point to data in input, x.
    // This way we can avoid making a copy of the same data.
    let x = ctx.get_input_mut(0)?;
    let y_dtype = *x.dtype();
    let data = x
        ._take_data()
        .ok_or_else(|| KernelError::Other("expected tensor to have data".to_string()))?;
    let y = ctx.get_output_mut(0)?;
    y._reshape(y_shape.to_vec().into_boxed_slice());
    y.init(data);
    y.set_dtype(y_dtype);
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
