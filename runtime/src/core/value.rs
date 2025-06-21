use crate::core::error::Error;
use half::f16;

#[derive(Debug)]
pub struct Value {
    pub(crate) shape: Option<Vec<usize>>,
    pub(crate) inner: InnerValue,
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum InnerValue {
    Float16(Vec<f16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Bool(Vec<bool>),
}

impl TryFrom<Value> for Vec<f16> {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value.inner {
            InnerValue::Float16(data) => Ok(data),
            _ => unimplemented!(),
        }
    }
}

impl TryFrom<Value> for Vec<f32> {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value.inner {
            InnerValue::Float32(data) => Ok(data),
            _ => unimplemented!(),
        }
    }
}

impl<T> TryFrom<Value> for (Vec<T>, Vec<usize>)
where
    Vec<T>: TryFrom<Value, Error = Error>,
{
    type Error = Error;

    fn try_from(mut value: Value) -> Result<Self, Self::Error> {
        // It's ok to take here because value gets dropped after this.
        let shape = value.shape.take().unwrap_or_default();
        let data = Vec::<T>::try_from(value)?;
        Ok((data, shape))
    }
}

impl TryFrom<Value> for Vec<i32> {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value.inner {
            InnerValue::Int32(data) => Ok(data),
            _ => unimplemented!(),
        }
    }
}

impl TryFrom<Value> for Vec<i64> {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value.inner {
            InnerValue::Int64(data) => Ok(data),
            _ => unimplemented!(),
        }
    }
}

impl TryFrom<Value> for Vec<bool> {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value.inner {
            InnerValue::Bool(data) => Ok(data),
            _ => unimplemented!(),
        }
    }
}

impl<'a, I> From<(Vec<I>, &'a [usize])> for Value
where
    Value: From<Vec<I>>,
{
    fn from(value: (Vec<I>, &'a [usize])) -> Self {
        (value.0, value.1.to_vec()).into()
    }
}

impl<I> From<(Vec<I>, Vec<usize>)> for Value
where
    Value: From<Vec<I>>,
{
    fn from(value: (Vec<I>, Vec<usize>)) -> Self {
        let mut res: Value = value.0.into();
        res.shape = Some(value.1);
        res
    }
}

impl From<Vec<f16>> for Value {
    fn from(value: Vec<f16>) -> Self {
        Self {
            inner: InnerValue::Float16(value),
            shape: None,
        }
    }
}

impl From<Vec<f32>> for Value {
    fn from(value: Vec<f32>) -> Self {
        Self {
            inner: InnerValue::Float32(value),
            shape: None,
        }
    }
}

impl From<Vec<i32>> for Value {
    fn from(value: Vec<i32>) -> Self {
        Self {
            inner: InnerValue::Int32(value),
            shape: None,
        }
    }
}

impl From<Vec<i64>> for Value {
    fn from(value: Vec<i64>) -> Self {
        Self {
            inner: InnerValue::Int64(value),
            shape: None,
        }
    }
}

impl From<Vec<bool>> for Value {
    fn from(value: Vec<bool>) -> Self {
        Self {
            inner: InnerValue::Bool(value),
            shape: None,
        }
    }
}
