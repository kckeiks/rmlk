use crate::core::allocators::ArenaId;
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
    BufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    Device {
        error: rmlk_cuda::Error,
    },
    ExecutionState(String),
    InvalidTensorIndex {
        index: usize,
    },
    InvalidAttribute {
        name: String,
    },
    InvalidAttributeDataType {
        name: String,
    },
    InvalidTensorShape {
        shape: Vec<usize>,
    },
    InvalidMemoryAllocation {
        message: String,
    },
    IncompatibleTensorShape {
        shapes: HashMap<usize, Vec<usize>>,
        op: Op,
    },
    InvalidAxis {
        axis: i32,
    },
    MissingDeviceData,
    MissingAttributes,
    MissingAttribute {
        name: String,
    },
    ExpectedShapeInDef {
        node_id: usize,
    },
    ExpectedDataTypeInDef {
        node_id: usize,
    },
    TensorStore(String),
    TensorNotFound {
        node_id: usize,
    },
    TensorNotFoundFromIndex {
        id: usize,
    },
    TensorIndexNotFound {
        node_id: usize,
    },
    UnableToConvertValue,
    UnexpectedTensorDataType {
        expected: DataType,
    },
    UnknownShapeBuffer {
        index: ArenaId,
    },
    UnsupportedDataType {
        dtype: DataType,
    },
    UnsupportedOp {
        op: Op,
    },
    UnsupportedOpForDataType {
        op: Op,
        dtype: DataType,
    },
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
            InternalError::ExpectedDataTypeInDef { node_id } => {
                write!(f, "expected data type in node `{node_id}`")
            }
            InternalError::ExpectedShapeInDef { node_id } => {
                write!(f, "expected shape in node `{node_id}`")
            }
            InternalError::Device { error: msg } => {
                write!(f, "device error `{msg:?}`")
            }
            InternalError::TensorNotFound { node_id } => {
                write!(f, "failed to find tensor: {node_id:?}")
            }
            InternalError::TensorNotFoundFromIndex { id } => {
                write!(f, "failed to find tensor from index: {id:?}")
            }
            InternalError::TensorIndexNotFound { node_id } => {
                write!(f, "failed to find tensor index: {node_id:?}")
            }
            InternalError::IncompatibleTensorShape { shapes, op } => {
                write!(f, "incompatible shapes `{shapes:?}` for {op:?}")
            }
            InternalError::InvalidTensorIndex { index } => {
                write!(f, "invalid tensor index: {index:?}")
            }
            InternalError::UnableToConvertValue => {
                write!(f, "unable to convert value")
            }
            InternalError::MissingDeviceData => {
                write!(f, "missing device data")
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
            InternalError::InvalidMemoryAllocation { message } => {
                write!(f, "invalid memory allocation `{message}`")
            }
            InternalError::BufferSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "expected buffer of size `{expected}` instead of `{actual}`"
                )
            }
            InternalError::UnknownShapeBuffer { index } => {
                write!(f, "unknown buffer given index `{:?}`", index)
            }
        }
    }
}

impl From<rmlk_cuda::Error> for InternalError {
    fn from(value: rmlk_cuda::Error) -> Self {
        Self::Device { error: value }
    }
}
