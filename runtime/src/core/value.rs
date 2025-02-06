use crate::core::error::Error;

#[derive(Debug)]
pub struct Value {
    pub(crate) inner: InnerValue,
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum InnerValue {
    Int32(Vec<i32>),
    Float32(Vec<f32>),
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

impl TryFrom<Value> for Vec<i32> {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value.inner {
            InnerValue::Int32(data) => Ok(data),
            _ => unimplemented!(),
        }
    }
}

impl From<Vec<f32>> for Value {
    fn from(value: Vec<f32>) -> Self {
        Self {
            inner: InnerValue::Float32(value),
        }
    }
}

impl From<Vec<i32>> for Value {
    fn from(value: Vec<i32>) -> Self {
        Self {
            inner: InnerValue::Int32(value),
        }
    }
}
