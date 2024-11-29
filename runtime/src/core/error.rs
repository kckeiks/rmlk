use crate::Value;
use rmlk_schema::{DataType, Op};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, InternalError>;

#[derive(Debug)]
pub enum Error {
    Internal { error: InternalError },
    ModelDeserializationFailed,
    NodeNotFound { id: usize },
    ExpectedName { node_id: usize },
    InvalidUserInput { input: HashMap<String, Value> },
    FailedToFindNodeId { name: String },
}

impl From<InternalError> for Error {
    fn from(value: InternalError) -> Self {
        Self::Internal { error: value }
    }
}

#[derive(Debug)]
pub enum InternalError {
    TensorStore(String),
    ExecutionState(String),
    Device { error: rmlk_cuda::Error },
    TensorNotFound { id: usize },
    TensorIndexNotFound { node_id: usize },
    InvalidTensorIndex { index: usize },
    MissingAttributes,
    MissingAttribute { name: String },
    InvalidAttribute { name: String },
    InvalidAttributeDataType { name: String },
    MissingData,
    UnableToConvertValue,
    UnexpectedTensorDataType { expected: DataType },
    UnsupportedDataType { dtype: DataType },
    UnsupportedOp { op: Op },
    UnsupportedOpForDataType { op: Op, dtype: DataType },
    InvalidTensorShape { shape: Vec<usize> },
    InvalidAxis { axis: i32 },
}

impl Display for InternalError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalError::TensorStore(msg) => {
                write!(f, "tensor store error `{msg}`")
            }
            InternalError::ExecutionState(msg) => {
                write!(f, "execution state error `{msg}`")
            }
            InternalError::Device { error: msg } => {
                write!(f, "device error `{msg:?}`")
            }
            InternalError::TensorNotFound { id } => {
                write!(f, "failed to find tensor: {id:?}")
            }
            InternalError::TensorIndexNotFound { node_id } => {
                write!(f, "failed to find tensor index: {node_id:?}")
            }
            InternalError::InvalidTensorIndex { index } => {
                write!(f, "invalid tensor index: {index:?}")
            }
            InternalError::UnableToConvertValue => {
                write!(f, "unable to convert value")
            }
            InternalError::MissingData => {
                write!(f, "missing data")
            }
            InternalError::MissingAttributes => {
                write!(f, "missing attributes")
            }
            InternalError::MissingAttribute { name } => {
                write!(f, "missing `{name}` attribute")
            }
            InternalError::InvalidAttribute { name } => {
                write!(f, "invalid attribute `{name}`")
            }
            InternalError::InvalidAttributeDataType { name } => {
                write!(f, "invalid data type for `{name}` attribute")
            }
            InternalError::UnexpectedTensorDataType { expected } => {
                write!(
                    f,
                    "unexpected tensor data type when expected `{expected:?}`"
                )
            }
            InternalError::UnsupportedDataType { dtype } => {
                write!(f, "unsupported `{dtype:?}` data type")
            }
            InternalError::UnsupportedOp { op } => {
                write!(f, "unsupported `{op:?}` op")
            }
            InternalError::UnsupportedOpForDataType { op, dtype } => {
                write!(f, "unsupported data type `{dtype:?}` for op `{op:?}`")
            }
            InternalError::InvalidTensorShape { shape } => {
                write!(f, "invalid tensor shape `{shape:?}`")
            }
            InternalError::InvalidAxis { axis } => {
                write!(f, "invalid axis `{axis}`")
            }
        }
    }
}

impl From<rmlk_cuda::Error> for InternalError {
    fn from(value: rmlk_cuda::Error) -> Self {
        Self::Device { error: value }
    }
}
