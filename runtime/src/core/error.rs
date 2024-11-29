use crate::Value;
use rmlk_schema::{DataType, Op};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, InternalError>;

#[derive(Debug)]
pub enum Error {
    Internal(InternalError),
    ModelDeserializationFailed,
    NodeNotFound { id: usize },
    ExpectedName { node_id: usize },
    InvalidUserInput { input: HashMap<String, Value> },
    FailedToFindNodeId { name: String },
}

impl From<InternalError> for Error {
    fn from(value: InternalError) -> Self {
        Self::Internal(value)
    }
}

#[derive(Debug)]
pub enum InternalError {
    TensorStore(String),
    ExecutionState(String),
    Device(rmlk_cuda::Error),
    TensorNotFound(usize),
    TensorIndexNotFound(usize),
    InvalidTensorIndex(usize),
    UnableToConvertValue,
    MissingAttributes,
    MissingData,
    UnexpectedTensorDataType { expected: DataType },
    UnsupportedDataType { dtype: DataType },
    UnsupportedOp { op: Op },
    UnsupportedOpForDataType { op: Op, dtype: DataType },
    InvalidTensorShape(Vec<usize>),
    InvalidAxis(i32),
}

impl Display for InternalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalError::TensorStore(msg) => {
                write!(f, "tensor store error: {msg}")
            }
            InternalError::ExecutionState(msg) => {
                write!(f, "execution state error: {msg}")
            }
            InternalError::Device(msg) => {
                write!(f, "device error: {msg:?}")
            }
            InternalError::TensorNotFound(id) => {
                write!(f, "failed to find tensor: {id:?}")
            }
            InternalError::TensorIndexNotFound(id) => {
                write!(f, "failed to find tensor index: {id:?}")
            }
            InternalError::InvalidTensorIndex(id) => {
                write!(f, "invalid tensor index: {id:?}")
            }
            InternalError::UnableToConvertValue => {
                write!(f, "unable to convert value")
            }
        }
    }
}

impl From<rmlk_cuda::Error> for InternalError {
    fn from(value: rmlk_cuda::Error) -> Self {
        Self::Device(value)
    }
}
