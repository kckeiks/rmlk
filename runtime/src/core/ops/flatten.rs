use crate::core::context::Context;
use crate::core::error::{Error, Result};

pub struct FlattenOp(());

impl FlattenOp {
    pub fn new() -> Self {
        Self(())
    }
    pub fn compute<T>(self, ctx: &mut Context<T>) -> Result<()> {
        _compute(ctx)
    }
}

pub fn _compute<T>(ctx: &mut Context<T>) -> Result<()> {
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
    y._reshape(y_shape.to_vec());
    // Todo: This tensor needs to point to data in input, x.
    // This way we can avoid making a copy of the same data.
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::core::context::Context;
    use crate::core::ops::flatten::_compute;
    use crate::core::test_utils;
    use crate::core::test_utils::{TestNode, TestParams};
    use rmlk_ir::{DataType, Op};

    #[test]
    fn test_flatten_f32() {
        let shape = vec![1, 1, 4, 4];
        let dtype = DataType::Float;

        let node_a = TestNode {
            shape,
            dtype,
            data: Some(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        };

        let node_c = TestNode {
            shape: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype,
            data: None,
        };

        let params = TestParams {
            inputs: vec![node_a],
            outputs: vec![node_c],
            attributes: Vec::new(),
            op: Op::Flatten,
        };

        let mut state = test_utils::build_graph_and_state(params);
        let mut context = Context::new(&mut state, 1).unwrap();

        _compute(&mut context).unwrap();

        let shape = context.get_output(0).unwrap().shape();

        assert_eq!(shape, &vec![1, 16])
    }
}
